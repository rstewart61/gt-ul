#!/bin/bash

LEFT=plots/"Dexter/DEFAULT/KM/elbow_silhouette.png"
RIGHT=plots/"Dexter Like Noise/DEFAULT/KM/elbow_silhouette.png"
convert +append "$LEFT" "$RIGHT" report/"Dexter_vs_Noise_KM.png"

for dr in DT ICA PCA RP; do
    for dataset in "Polish Bankruptcy" Dexter; do
        LEFT=plots/"${dataset}/${dr}/KM/KM[$dr] - Cluster labels_best_tsne.png"
        MIDDLE=plots/"${dataset}/${dr}/DEFAULT/Ground truth_best_tsne.png"
        RIGHT=plots/"${dataset}/${dr}/EM/EM[$dr] - Cluster labels_best_tsne.png"
        convert +append "$LEFT" "$MIDDLE" "$RIGHT" report/"$dataset $dr TSNE comparison.png"

        LEFT=plots/"Polish Bankruptcy/$dr/EM/ch_vs_db.png"
        RIGHT=plots/"Dexter/$dr/EM/ch_vs_db.png"
        convert +append "$LEFT" "$RIGHT" report/"Both_${dr}_EM_ch_vs_db.png"
        
        LEFT=plots/"${dataset}/$dr/KM/elbow_silhouette.png"
        RIGHT=plots/"${dataset}/$dr/KM/Largest Cluster Size % of Samples.png"
        convert +append "$LEFT" "$RIGHT" report/"${dataset}_${dr}_silhoutte_vs_cluster_size.png"
    done
done

for dataset in "Polish Bankruptcy" Dexter; do
    LEFT=plots/"${dataset}/DEFAULT/KM/KM - Cluster labels_best_tsne.png"
    MIDDLE=plots/"${dataset}/DEFAULT/DEFAULT/Ground truth_best_tsne.png"
    RIGHT=plots/"${dataset}/DEFAULT/EM/EM - Cluster labels_best_tsne.png"
    convert +append "$LEFT" "$MIDDLE" "$RIGHT" report/"$dataset DEFAULT TSNE comparison.png"
    
    ONE=plots/"${dataset}/RP/KM/elbow_silhouette.png"
    TWO=plots/"${dataset}/PCA/KM/elbow_silhouette.png"
    THREE=plots/"${dataset}/ICA/KM/elbow_silhouette.png"
    FOUR=plots/"${dataset}/DT/KM/elbow_silhouette.png"
    convert +append "$ONE" "$TWO" "$THREE" "$FOUR" report/"$dataset KM Elbow Silhouette comparison.png"

    ONE=plots/"${dataset}/RP/KM/Largest Cluster Size % of Samples.png"
    TWO=plots/"${dataset}/PCA/KM/Largest Cluster Size % of Samples.png"
    THREE=plots/"${dataset}/ICA/KM/Largest Cluster Size % of Samples.png"
    FOUR=plots/"${dataset}/DT/KM/Largest Cluster Size % of Samples.png"
    convert +append "$ONE" "$TWO" "$THREE" "$FOUR" report/"$dataset KM Largest Cluster Size % of Samples comparison.png"
done

LEFT=plots/"Polish Bankruptcy/DEFAULT/PCA/explained_variance_ratio.png"
RIGHT=plots/"Dexter/DEFAULT/PCA/explained_variance_ratio.png"
convert +append "$LEFT" "$RIGHT" report/"Both_PCA_explained_variance_ratio.png"

LEFT=plots/"Polish Bankruptcy/DEFAULT/RP/reconstruction_error.png"
RIGHT=plots/"Dexter/DEFAULT/RP/reconstruction_error.png"
convert +append "$LEFT" "$RIGHT" report/"Both_RP_reconstruction_error.png"

LEFT=plots/"Polish Bankruptcy/DEFAULT/RP/johnson-lindenstrauss.png"
RIGHT=plots/"Dexter/DEFAULT/RP/johnson-lindenstrauss.png"
convert +append "$LEFT" "$RIGHT" report/"Both_RP_johnson-lindenstrauss.png"

LEFT=plots/"Polish Bankruptcy/DEFAULT/ICA/kurtosis.png"
RIGHT=plots/"Dexter/DEFAULT/ICA/kurtosis.png"
convert +append "$LEFT" "$RIGHT" report/"Both_ICA_kurtosis.png"

LEFT=plots/"Polish Bankruptcy/DEFAULT/DT/feature_importances.png"
RIGHT=plots/"Dexter/DEFAULT/DT/feature_importances.png"
convert +append "$LEFT" "$RIGHT" report/"Both_DT_feature_importances.png"

for dataset in "Polish Bankruptcy" Dexter; do
    LEFT=plots/"${dataset}/DEFAULT/DEFAULT/nn_results_Balanced Accuracy.png"
    RIGHT=plots/"${dataset}/DEFAULT/DEFAULT/nn_results_Tuning Time (s).png"
    convert +append "$LEFT" "$RIGHT" report/"${dataset}_nn_results.png"

    LEFT=plots/"${dataset}/DEFAULT/KM/DR comparison for KM - Silhouette.png"
    RIGHT=plots/"${dataset}/DEFAULT/KM/DR comparison for KM - Largest Cluster Size % of Samples.png"
    convert +append "$LEFT" "$RIGHT" report/"${dataset}_KM_Silhouette Comparison.png"

    PCA=plots/"${dataset}/DEFAULT/PCA/PCA Features 1, 2 for ${dataset}.png"
    ICA=plots/"${dataset}/DEFAULT/ICA/ICA Features 1, 2 for ${dataset}.png"
    RP=plots/"${dataset}/DEFAULT/RP/RP Features 1, 2 for ${dataset}.png"
    DT=plots/"${dataset}/DEFAULT/DT/DT Features 1, 2 for ${dataset}.png"
    convert +append "$PCA" "$ICA"  "$RP" "$DT" report/"${dataset}_dimensions_1_2.png"
done

LEFT=plots/"Polish Bankruptcy/DEFAULT/DEFAULT/cluster_results_ami.png"
RIGHT=plots/"Dexter/DEFAULT/DEFAULT/cluster_results_ami.png"
convert +append "$LEFT" "$RIGHT" report/"Both_cluster_results_ami.png"

echo Finished gluing plots into report directory
