# Author: Devon Campbell (dec2180)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, f1_score, det_curve, precision_recall_curve
import logging
import os
#from utils import XvectorDataset, txt_to_df, get_protocol, load_xvectors, link_xvectors_and_labels, get_xvector_and_labels
from utils_example import XvectorDataset, txt_to_df, get_protocol, load_xvectors, link_xvectors_and_labels, get_xvector_and_labels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_centroid(embeddings):
    return torch.mean(embeddings, dim=0)

def cosine_similarity(X, c):
    # Normalize the vectors to unit length
    X_norm = X / torch.norm(X, p=2, dim=1, keepdim=True)
    c_norm = c / torch.norm(c, p=2)
    # Dot product of normalized vectors
    return torch.matmul(X_norm, c_norm.unsqueeze(-1)).squeeze(-1)

def euclidean_similarity(X, c):
    # Calculate the Euclidean distance between X and c
    distances = torch.cdist(X, c.unsqueeze(0))
    # Convert distances to similarities
    similarities = 1 / (1 + distances)
    return similarities.squeeze()

def manhattan_similarity(X, c):
    # Calculate the Manhattan distance between X and c
    distances = torch.norm(X - c, p=1, dim=1)
    # Convert distances to similarities
    similarities = 1 / (1 + distances)
    return similarities

def minkowski_similarity(X, c, p=2):
    # Calculate the Minkowski distance between X and c
    distances = torch.norm(X - c, p=p, dim=1)
    # Convert distances to similarities
    similarities = 1 / (1 + distances)
    return similarities

def classify_audio(X, centroid, metric, threshold=0.5):
    if metric=='cosine':
        similarities = cosine_similarity(X.xvectors, centroid)
    elif metric=='euclidean':
        similarities = euclidean_similarity(X.xvectors, centroid)
    elif metric=='minkowski':
        similarities = minkowski_similarity(X.xvectors, centroid)
    else:
         similarities = manhattan_similarity(X.xvectors, centroid)

    # Calculate median similarity for each unique label
    unique_labels = torch.unique(X.labels)
    label_similarities = {}

    for label in unique_labels:
        # Get similarities for the current label
        label_mask = (X.labels == label)
        label_similarities[label.item()] = similarities[label_mask].numpy()
    return (similarities > threshold).long(), label_similarities

def calculate_accuracy(labels, predictions):
    correct_predictions = torch.sum(labels == predictions)
    accuracy = correct_predictions.float() / len(labels)
    return accuracy.item()

def calculate_f1_score(labels, predictions):
    labels_np = labels.cpu().detach().numpy() if labels.is_cuda else labels.numpy()
    predictions_np = predictions.cpu().detach().numpy() if predictions.is_cuda else predictions.numpy()

    labels_np = labels_np.astype(int)
    predictions_np = predictions_np.astype(int)

    return f1_score(labels_np, predictions_np, average='binary')

def optimize_threshold_by_precision_recall(train_dataset, reference_embeddings, metric):
    scores, labels = get_scores_and_labels(train_dataset, reference_embeddings, metric)
    precision, recall, thresholds = precision_recall_curve(labels.numpy(), scores.numpy())
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)  # Avoid division by zero
    max_index = np.argmax(f1_scores)  # Find index of maximum F1 score
    return thresholds[max_index], f1_scores[max_index]

def adjust_threshold_based_on_dev(dev_dataset, reference_embeddings, initial_threshold, metric):
    current_best_threshold = initial_threshold  # Start with the initial threshold from training
    current_best_f1 = 0
    dev_sims = []

    # Define a range around the initial threshold to fine-tune
    threshold_range = np.linspace(max(0, initial_threshold - 0.2), min(1, initial_threshold + 0.2), 41)

    for threshold in threshold_range:
        predictions, dev_sims = classify_audio(dev_dataset, compute_centroid(reference_embeddings), metric, threshold)
        f1 = calculate_f1_score(dev_dataset.labels, predictions)
        if f1 > current_best_f1:
            current_best_f1 = f1
            current_best_threshold = threshold
            #logging.info(f"Updated best threshold to {threshold} with F1 score {f1}")

    return current_best_threshold, dev_sims

def get_scores_and_labels(dataset, embeddings, metric):
    # Assuming a function that computes similarities and gets labels
    centroid = compute_centroid(embeddings)
    if metric=='cosine':
        similarities = cosine_similarity(dataset.xvectors, centroid)
    elif metric=='euclidean':
        similarities = euclidean_similarity(dataset.xvectors, centroid)
    elif metric=='minkowski':
        similarities = minkowski_similarity(dataset.xvectors, centroid)
    else:
         similarities = manhattan_similarity(dataset.xvectors, centroid)
    return similarities, dataset.labels

def calculate_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer, eer_threshold

def plot_det_curve(labels, scores,  metric):
    fpr, fnr, thresholds = det_curve(labels, scores)
    plt.figure()
    plt.plot(fpr, fnr)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title(f'{metric.capitalize()} Detection Error Tradeoff Curve')
    plt.grid(True)
    plt.savefig(f'./graphs/{metric}_det.png')

def plot_label_similarities(label_similarities, phase, metric):
    plt.figure(figsize=(10, 6))
    for label, sims in label_similarities.items():
        plt.hist(sims, bins=30, alpha=0.5, label=f'Label {label}')

    plt.xlabel(f'{metric.capitalize()}')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {metric.capitalize()} for {phase.capitalize()} set')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./graphs/{metric}_{phase}_similarity_distribution.png')

def max_similarity(X, reference_set, batch_size=100):
    # Normalize the reference vectors
    ref_norm = reference_set / torch.norm(reference_set, p=2, dim=1, keepdim=True)
    logging.info(f"Normalized reference embeddings. Shape: {ref_norm.shape}")

    # Initialize the maximum similarities with very small numbers
    max_similarities = torch.full((X.xvectors.shape[0],), -float('inf'), device=X.xvectors.device)
    logging.info(f"Initialized maximum similarities tensor.")

    # Process in batches
    num_batches = (X.xvectors.shape[0] + batch_size - 1) // batch_size
    logging.info(f"Starting batch processing for max similarity calculation. Total batches: {num_batches}")
    
    for i, start_idx in enumerate(range(0, X.xvectors.shape[0], batch_size)):
        end_idx = start_idx + batch_size
        X_batch = X.xvectors[start_idx:end_idx]

        # Normalize the batch of test vectors
        X_norm = X_batch / torch.norm(X_batch, p=2, dim=1, keepdim=True)

        # Compute cosine similarities and find the max for each batch
        batch_similarities = torch.matmul(X_norm, ref_norm.transpose(0, 1))
        max_batch_similarities = torch.max(batch_similarities, dim=1)[0]

        # Update the global max similarities
        max_similarities[start_idx:end_idx] = max_batch_similarities

        if i%50==0:
            logging.info(f"Processed batch {i+1}/{num_batches}: Indices {start_idx} to {end_idx}")

    logging.info("Completed all batches for max similarity calculation.")
    return max_similarities

def classify_audio_with_max_sim(X, reference_set, threshold=0.5):
    max_similarities = max_similarity(X, reference_set)

    unique_labels = torch.unique(X.labels)
    label_similarities = {}

    for label in unique_labels:
        # Get similarities for the current label
        label_mask = (X.labels == label)
        label_similarities[label.item()] = max_similarities[label_mask].numpy()
    return (max_similarities > threshold).long(), label_similarities

def main():
    logging.info("Beginning centroid-based analysis")
    datasets = {
        "train": XvectorDataset(get_xvector_and_labels("train")),
        "dev": XvectorDataset(get_xvector_and_labels("dev")),
        "eval": XvectorDataset(get_xvector_and_labels("eval"))
    }

    xvector_dir_path = './exp/xvector_nnet_1a/xvectors_train'
    #xvector_dir_path = './example_xvectors/reference/voxceleb'
    reference_embeddings = load_xvectors(xvector_dir_path)
    logging.info("Retreived all embeddings & labels")

    metrics = ['cosine','euclidean', 'minkowski', 'manhattan','ms']
    for metric in metrics:
        logging.info(f"\t{metric.capitalize()} analysis")
        for phase, dataset in datasets.items():
            logging.info(f"Begin {metric} {phase} analysis...")
            # Compute the centroid of reference embeddings
            centroid = compute_centroid(reference_embeddings)
            
            # Optimize the threshold and evaluate based on the phase
            if phase == "train":
                optimal_threshold, best_f1 = optimize_threshold_by_precision_recall(dataset, reference_embeddings, metric)
                logging.info(f"\t{phase.capitalize()} optimal threshold: {optimal_threshold} with F1 score: {best_f1}")
            elif phase == "dev":
                optimal_threshold, sims = adjust_threshold_based_on_dev(dataset, reference_embeddings, optimal_threshold, metric)
                logging.info(f"\tFine-tuned optimal threshold with {phase} dataset: {optimal_threshold}")

            if metric != 'ms':
                # Classify audio using the optimized threshold
                logging.info("\tCalculating centroid predictions...")
                predictions, sims = classify_audio(dataset, centroid, metric, optimal_threshold)
            else:
                logging.info("\tCalculating max predictions...")
                predictions, sims = classify_audio_with_max_sim(dataset, reference_embeddings, optimal_threshold)

            logging.info("\tBegin evaluating performance...")
            # Calculate accuracy and F1 score
            accuracy = calculate_accuracy(dataset.labels, predictions)
            f1_score = calculate_f1_score(dataset.labels, predictions)
            logging.info(f"\t\t{phase.capitalize()} {metric} Accuracy: {accuracy}")
            logging.info(f"\t\t{phase.capitalize()} {metric} F1 Score: {f1_score}")

            # Special operations for the evaluation dataset
            if phase == "eval":
                scores, labels = get_scores_and_labels(dataset, reference_embeddings, metric)
                eer, eer_threshold = calculate_eer(labels, scores)
                logging.info(f"{metric} EER: {eer} at Threshold: {eer_threshold}")
                plot_det_curve(labels, scores, metric)

            # Plot similarities for each phase
            plot_label_similarities(sims, phase, metric)

if __name__ == "__main__":
    main()