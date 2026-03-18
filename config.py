''' Configuration File.
'''

CUDA_VISIBLE_DEVICES = 0
NUM_TRAIN = 50000 # N \Fashion MNIST 60000, cifar 10/100 50000
# NUM_VAL   = 50000 - NUM_TRAIN
BATCH     = 128 # B
SUBSET    = 10000 # M
ADDENDUM  = 5000 # K

MARGIN = 1.0 # xi
WEIGHT = 1.0 # lambda
ADV_WT = 0.1
FEAT_WT = 1
LOGIT_WT = 1

TRIALS = 3
CYCLES = 10

EPOCH = 200
EPOCH_GCN = 200
LR = 0.01
LR_GCN = 1e-3
LR_DECODER = 0.01
LR_SAMPLER = 0.01
MILESTONES = [80, 150]
EPOCHL = 120#20 #120 # After 120 epochs, stop 
EPOCHV = 100 # VAAL number of epochs

MOMENTUM = 0.9
WDECAY = 5e-4  # 5e-4

# Teacher checkpoints (relative to project root; configurable by user)
TEACHER_PATH_C10 = 'checkpoints/cifar10_vgg19_teacher.pth'
TEACHER_PATH_C100 = 'checkpoints/cifar100_vgg19_teacher.pth'
SAMPLER_STEPS = 1
EPOCH_FDAL = 20