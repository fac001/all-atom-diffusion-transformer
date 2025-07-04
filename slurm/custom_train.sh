#!/bin/bash

eval "$(mamba shell hook --shell bash)"
mamba activate adit

############################################################

#! Work directory (i.e. where the job will run):
workdir="~/workingproj/adit/all-atom-diffusion-transformer"

#! Full path to application executable:
application="python $workdir/src/train_autoencoder.py"

#! Set hparams in configs/autoencoder_module/vae.yaml, or below:
latent_dim=8    # 4 / 8
loss_kl=0.00001 # 0.0001 / 0.00001

#! (for logging purposes)
latent_str="latent@${latent_dim}"
kl_str="kl@${loss_kl}"
name="vae_${latent_str}_${kl_str}"

#! Run options for the application:
options="trainer=ddp logger=wandb name=$name ++autoencoder_module.latent_dim=$latent_dim ++autoencoder_module.loss_weights.loss_kl.mp20=$loss_kl ++autoencoder_module.loss_weights.loss_kl.qm9=$loss_kl"

CMD="$application $options"

###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to $(pwd).\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: $(date)"
echo "Running on master node: $(hostname)"
echo "Current directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD
