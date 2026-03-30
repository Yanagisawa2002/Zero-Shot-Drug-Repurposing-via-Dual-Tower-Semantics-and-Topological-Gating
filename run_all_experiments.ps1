$ErrorActionPreference = 'Stop'

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $RootDir

$Seeds = @(42, 123, 2026)
$Splits = @('random', 'cross_drug', 'cross_disease')
$Variants = @('baseline_pure_gnn', 'single_tower', 'sota_dual_tower', 'ablation_no_gnn', 'variant_path_loss')

$LogDir = Join-Path $RootDir 'outputs/final_ablation_logs_20260329'
$RunDir = Join-Path $RootDir 'outputs/final_ablation_runs_20260329'
$EvalDir = Join-Path $RootDir 'outputs/final_ablation_pairfixed_ho_20260329'

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
New-Item -ItemType Directory -Force -Path $EvalDir | Out-Null

$TrainScript = Join-Path $RootDir 'scripts/train_quad_split_ho_probe.py'
$HoEvalScript = Join-Path $RootDir 'scripts/eval_pair_fixed_ho.py'

$FeatureDir = Join-Path $RootDir 'outputs/pubmedbert_hybrid_features_clean'
$TripletTextPkl = Join-Path $RootDir 'triplet_text_embeddings.pkl'
$DrugMorganPkl = Join-Path $RootDir 'drug_morgan_fingerprints.pkl'
$DrugTextPkl = Join-Path $RootDir 'thick_drug_text_embeddings_sapbert.pkl'
$DiseaseTextPkl = Join-Path $RootDir 'thick_disease_text_embeddings_sapbert.pkl'
$NodesCsv = Join-Path $RootDir 'data/PrimeKG/nodes.csv'
$EdgesCsv = Join-Path $RootDir 'data/PrimeKG/edges.csv'

function Get-ProcessedPath {
    param([string]$Split)
    switch ($Split) {
        'random' { return (Join-Path $RootDir 'data/PrimeKG/processed/primekg_indication_mvp.pt') }
        'cross_drug' { return (Join-Path $RootDir 'data/PrimeKG/processed/primekg_indication_cross_drug.pt') }
        'cross_disease' { return (Join-Path $RootDir 'data/PrimeKG/processed/primekg_indication_cross_disease.pt') }
        default { throw "Unknown split: $Split" }
    }
}

function Get-OtCsv {
    param([string]$Split)
    switch ($Split) {
        'random' { return (Join-Path $RootDir 'outputs/ot_random_external_profile_pair_clean/novel_ood_triplets.csv') }
        'cross_drug' { return (Join-Path $RootDir 'outputs/ot_cross_drug_external_profile_pair_clean/novel_ood_triplets.csv') }
        'cross_disease' { return (Join-Path $RootDir 'outputs/ot_cross_disease_external_profile_pair_clean/novel_ood_triplets.csv') }
        default { throw "Unknown split: $Split" }
    }
}

function Get-VariantFlags {
    param([string]$Variant)

    switch ($Variant) {
        'baseline_pure_gnn' {
            return @(
                '--disable-disease-semantic',
                '--path-loss-weight', '0.0'
            )
        }
        'single_tower' {
            return @(
                '--disease-text-embeddings-path', $DiseaseTextPkl,
                '--path-loss-weight', '0.0'
            )
        }
        'sota_dual_tower' {
            return @(
                '--drug-text-embeddings-path', $DrugTextPkl,
                '--disease-text-embeddings-path', $DiseaseTextPkl,
                '--path-loss-weight', '0.0'
            )
        }
        'ablation_no_gnn' {
            return @(
                '--drug-text-embeddings-path', $DrugTextPkl,
                '--disease-text-embeddings-path', $DiseaseTextPkl,
                '--ablate-gnn',
                '--path-loss-weight', '0.0'
            )
        }
        'variant_path_loss' {
            return @(
                '--drug-text-embeddings-path', $DrugTextPkl,
                '--disease-text-embeddings-path', $DiseaseTextPkl,
                '--path-loss-weight', '0.1'
            )
        }
        default { throw "Unknown variant: $Variant" }
    }
}

Write-Output '============================================================'
Write-Output 'Final ablation run started'
Write-Output "ROOT_DIR=$RootDir"
Write-Output "LOG_DIR=$LogDir"
Write-Output "RUN_DIR=$RunDir"
Write-Output "EVAL_DIR=$EvalDir"
Write-Output '============================================================'

foreach ($Split in $Splits) {
    try {
        $ProcessedPath = Get-ProcessedPath -Split $Split
        $OtCsv = Get-OtCsv -Split $Split
    }
    catch {
        Write-Output "[ERROR] Failed to resolve paths for split=$Split"
        Write-Output $_
        continue
    }

    foreach ($Variant in $Variants) {
        try {
            $VariantFlags = Get-VariantFlags -Variant $Variant
        }
        catch {
            Write-Output "[ERROR] Failed to build flags for variant=$Variant"
            Write-Output $_
            continue
        }

        foreach ($Seed in $Seeds) {
            $RunStem = "${Split}_${Variant}_seed${Seed}"
            $TrainLog = Join-Path $LogDir "${RunStem}_train.log"
            $HoLog = Join-Path $LogDir "${RunStem}_ho_eval.log"
            $OutputJson = Join-Path $RunDir "${RunStem}.json"
            $CheckpointPath = Join-Path $RunDir "${RunStem}.pt"
            $HoJson = Join-Path $EvalDir "${RunStem}.json"

            Write-Output ''
            Write-Output '------------------------------------------------------------'
            Write-Output "[RUN] split=$Split variant=$Variant seed=$Seed"
            Write-Output '------------------------------------------------------------'

            $TrainArgs = @(
                $TrainScript,
                '--processed-path', $ProcessedPath,
                '--output-json', $OutputJson,
                '--checkpoint-path', $CheckpointPath,
                '--nodes-csv', $NodesCsv,
                '--edges-csv', $EdgesCsv,
                '--feature-dir', $FeatureDir,
                '--triplet-text-embeddings-path', $TripletTextPkl,
                '--drug-morgan-fingerprints-path', $DrugMorganPkl,
                '--text-distill-alpha', '0.2',
                '--primary-loss-type', 'bce',
                '--epochs', '60',
                '--batch-size', '32',
                '--hidden-channels', '128',
                '--out-dim', '128',
                '--scorer-hidden-dim', '128',
                '--lr', '1e-3',
                '--weight-decay', '1e-5',
                '--dropout', '0.1',
                '--initial-residual-alpha', '0.2',
                '--encoder-type', 'rgcn',
                '--agg-type', 'attention',
                '--graph-surgery-mode', 'direct_only',
                '--use-early-external-fusion',
                '--dropedge-p', '0.15',
                '--seed', "$Seed",
                '--ot-novel-csv', $OtCsv
            ) + $VariantFlags

            Write-Output ('[TRAIN] python ' + ($TrainArgs -join ' '))
            try {
                & python @TrainArgs *> $TrainLog
            }
            catch {
                Write-Output "[ERROR] Training failed for split=$Split variant=$Variant seed=$Seed"
                Write-Output "        See log: $TrainLog"
                continue
            }

            if (-not (Test-Path -LiteralPath $CheckpointPath)) {
                Write-Output "[ERROR] Checkpoint missing after training: $CheckpointPath"
                continue
            }

            $HoArgs = @(
                $HoEvalScript,
                '--checkpoint-path', $CheckpointPath,
                '--processed-path', $ProcessedPath,
                '--nodes-csv', $NodesCsv,
                '--edges-csv', $EdgesCsv,
                '--feature-dir', $FeatureDir,
                '--ot-novel-csv', $OtCsv,
                '--graph-surgery-mode', 'direct_only',
                '--batch-size', '512',
                '--seed', "$Seed",
                '--output-json', $HoJson
            )
            if ($Variant -eq 'ablation_no_gnn') {
                $HoArgs += '--ablate-gnn'
            }

            Write-Output ('[HO-EVAL] python ' + ($HoArgs -join ' '))
            try {
                & python @HoArgs *> $HoLog
            }
            catch {
                Write-Output "[ERROR] Pair-fixed HO eval failed for split=$Split variant=$Variant seed=$Seed"
                Write-Output "        See log: $HoLog"
                continue
            }

            Write-Output "[DONE] split=$Split variant=$Variant seed=$Seed"
            Write-Output "       train_log=$TrainLog"
            Write-Output "       ho_log=$HoLog"
            Write-Output "       ckpt=$CheckpointPath"
            Write-Output "       json=$OutputJson"
            Write-Output "       ho_json=$HoJson"
        }
    }
}

Write-Output '============================================================'
Write-Output 'Final ablation run finished'
Write-Output "Logs:  $LogDir"
Write-Output "Runs:  $RunDir"
Write-Output "HO:    $EvalDir"
Write-Output '============================================================'
