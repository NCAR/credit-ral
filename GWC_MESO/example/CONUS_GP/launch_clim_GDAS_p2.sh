#!/bin/bash
#PBS -A P48500028
#PBS -N Pred_GDAS_p2
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=64:ngpus=1
#PBS -q main
#PBS -j oe
#PBS -k eod
#PBS -r n
# Load modules
module purge
module load ncarenv/24.12
module reset
module load gcc craype cray-mpich cuda cudnn conda
conda activate credit-new
# Export environment variables
export LSCRATCH=/glade/derecho/scratch/ksha/
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
export NCCL_SOCKET_IFNAME=hsn
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_IB_DISABLE=1
export NCCL_CROSS_NIC=1
export NCCL_NCHANNELS_PER_NET_PEER=4
export MPICH_RDMA_ENABLED_CUDA=1
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PBH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_OPTIMIZED_MRS=false
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
# Launch MPIs
nodes=( $( cat $PBS_NODEFILE ) )
echo nodes: $nodes
# Find headnode's IP:
head_node=${nodes[0]}
head_node_ip=$(host $head_node | awk '{print $NF}')
MASTER_ADDR=$head_node_ip MASTER_PORT=1234 mpiexec -n 1 --ppn 1 --cpu-bind none python /glade/u/home/ksha/credit-mini/applications/WRF_pred_future.py -c /glade/work/ksha/DWC_runs/CONUS_GP_clean/model_clim_GDAS_p2.yml
