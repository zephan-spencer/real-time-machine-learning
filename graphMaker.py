import os
# Supress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
from tbparse import SummaryReader

from matplotlib import rc
rc('mathtext', default='regular')

log_dir = "/home/he-man/real-time-machine-learning/runs"

dataReader = SummaryReader(os.path.join(log_dir), extra_columns={'dir_name'})
dataFrame = dataReader.scalars

# Problem 1A and 1B Stats

prob1ATrain = dataFrame[(dataFrame['dir_name'] == 'prob1A') & (dataFrame['tag'] == 'CIFAR-10 Training Accuracy')]
prob1AVal = dataFrame[(dataFrame['dir_name'] == 'prob1A') & (dataFrame['tag'] == 'CIFAR-10 Validation Accuracy')]
prob1ALoss = dataFrame[(dataFrame['dir_name'] == 'prob1A') & (dataFrame['tag'] == 'CIFAR-10 Training loss')]

prob1BTrain = dataFrame[(dataFrame['dir_name'] == 'prob1B') & (dataFrame['tag'] == 'CIFAR-10 Training Accuracy')]
prob1BVal = dataFrame[(dataFrame['dir_name'] == 'prob1B') & (dataFrame['tag'] == 'CIFAR-10 Validation Accuracy')]
prob1BLoss = dataFrame[(dataFrame['dir_name'] == 'prob1B') & (dataFrame['tag'] == 'CIFAR-10 Training loss')]

fig, ax1 = plt.subplots()
prob1ATrainLine = ax1.plot(prob1ATrain['step'], prob1ATrain['value'],  '-', color="#fb6f37", label='1A Train')
prob1AValLine = ax1.plot(prob1AVal['step'], prob1AVal['value'],  '--', color="#fb6f37", label='1A Val')
prob1BTrainLine = ax1.plot(prob1BTrain['step'], prob1BTrain['value'],  '-', color="#6dc5a9", label='1B Train')
prob1BValLine = ax1.plot(prob1BVal['step'], prob1BVal['value'],  '--', color="#6dc5a9", label='1B Val')

ax2 = ax1.twinx()
prob1ALossLine = ax2.plot(prob1ALoss['step'], prob1ALoss['value'],  ':', color="#fb6f37", label='1A Loss')
prob1BLossLine = ax2.plot(prob1BLoss['step'], prob1BLoss['value'],  ':', color="#6dc5a9", label='1B Loss')

lns = prob1ATrainLine + prob1AValLine + prob1ALossLine + prob1BTrainLine + prob1BValLine + prob1BLossLine

labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, bbox_to_anchor=(1,.6))

ax1.grid()
ax1.set_ylim([0, 100])
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy %')

ax2.set_ylabel('Loss')

plt.title('Problem 1A and 1B Statistics')
plt.savefig("figures/problem1A and 1B statistics.png", facecolor='w', dpi=300)

# Problem 2A Training, Validation, and Loss Graph

baselineTrain = dataFrame[(dataFrame['dir_name'] == 'prob2A') & (dataFrame['tag'] == 'CIFAR-10 Training Accuracy')]
baselineVal = dataFrame[(dataFrame['dir_name'] == 'prob2A') & (dataFrame['tag'] == 'CIFAR-10 Validation Accuracy')]
baselineLoss = dataFrame[(dataFrame['dir_name'] == 'prob2A') & (dataFrame['tag'] == 'CIFAR-10 Training loss')]

fig, ax1 = plt.subplots()
baselineTrainLine = ax1.plot(baselineTrain['step'], baselineTrain['value'],  '-', color="#fb6f37", label='BatchNorm Train')
baselineValLine = ax1.plot(baselineVal['step'], baselineVal['value'],  '-', color="#6dc5a9", label='BatchNorm Val')

ax2 = ax1.twinx()
baselineLossLine = ax2.plot(baselineLoss['step'], baselineLoss['value'],  '-', color="#738abf", label='BatchNorm Loss')

lns = baselineTrainLine + baselineValLine + baselineLossLine

labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, bbox_to_anchor=(1,.5))

ax1.grid()
ax1.set_ylim([0, 100])
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy %')

ax2.set_ylabel('Loss')

plt.title('Problem 2A Statistics')
plt.savefig("figures/problem2A statistics.png", facecolor='w', dpi=300)

# Problem 2B Training and Validation Accuracy Graph

batchNormTrain = dataFrame[(dataFrame['dir_name'] == 'prob2BBatchNorm') & (dataFrame['tag'] == 'CIFAR-10 Training Accuracy')]
batchNormVal = dataFrame[(dataFrame['dir_name'] == 'prob2BBatchNorm') & (dataFrame['tag'] == 'CIFAR-10 Validation Accuracy')]

dropoutTrain = dataFrame[(dataFrame['dir_name'] == 'prob2BDropout') & (dataFrame['tag'] == 'CIFAR-10 Training Accuracy')]
dropoutVal = dataFrame[(dataFrame['dir_name'] == 'prob2BDropout') & (dataFrame['tag'] == 'CIFAR-10 Validation Accuracy')]

weightDecayTrain = dataFrame[(dataFrame['dir_name'] == 'prob2BWeightDecay') & (dataFrame['tag'] == 'CIFAR-10 Training Accuracy')]
weightDecayVal = dataFrame[(dataFrame['dir_name'] == 'prob2BWeightDecay') & (dataFrame['tag'] == 'CIFAR-10 Validation Accuracy')]

fig, ax1 = plt.subplots()

batchNormTrainLine = ax1.plot(batchNormTrain['step'], batchNormTrain['value'],  '-', color="#fb6f37", label='BatchNorm Train')
batchNormValLine = ax1.plot(batchNormVal['step'], batchNormVal['value'],  '--', color="#fb6f37", label='BatchNorm Val')

dropoutTrainLine = ax1.plot(dropoutTrain['step'], dropoutTrain['value'],  '-', color="#6dc5a9", label='Dropout Train')
dropoutValLine = ax1.plot(dropoutVal['step'], dropoutVal['value'],  '--', color="#6dc5a9", label='Dropout Val')

weightDecayTrainLine = ax1.plot(weightDecayTrain['step'], weightDecayTrain['value'],  '-', color="#738abf", label='Weight Decay Train')
weightDecayValLine = ax1.plot(weightDecayVal['step'], weightDecayVal['value'],  '--', color="#738abf", label='Weight Decay Val')

lns = batchNormTrainLine + batchNormValLine + dropoutTrainLine + dropoutValLine + weightDecayTrainLine + weightDecayValLine

labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, bbox_to_anchor=(1,.5))

ax1.grid()
ax1.set_ylim([0, 100])
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy %')

plt.title('Problem 2B Training and Validation Accuracies')
plt.savefig("figures/Problem 2B Training and Validation Accuracies.png", facecolor='w', dpi=300)


# Problem 2B Loss Graph

batchNormLoss = dataFrame[(dataFrame['dir_name'] == 'prob2BBatchNorm') & (dataFrame['tag'] == 'CIFAR-10 Training loss')]
dropoutLoss = dataFrame[(dataFrame['dir_name'] == 'prob2BDropout') & (dataFrame['tag'] == 'CIFAR-10 Training loss')]
weightDecayLoss = dataFrame[(dataFrame['dir_name'] == 'prob2BWeightDecay') & (dataFrame['tag'] == 'CIFAR-10 Training loss')]

fig, ax1 = plt.subplots()

batchNormLossLine = ax1.plot(batchNormLoss['step'], batchNormLoss['value'],  '-', color="#fb6f37", label='BatchNorm Loss')
dropoutLossLine = ax1.plot(dropoutLoss['step'], dropoutLoss['value'],  '-', color="#6dc5a9", label='Dropout Loss')
weightDecayLossLine = ax1.plot(weightDecayLoss['step'], weightDecayLoss['value'],  '-', color="#738abf", label='Weight Decay Loss')

lns = batchNormLossLine + dropoutLossLine + weightDecayLossLine

labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, bbox_to_anchor=(1,.5))

ax1.grid()
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy %')

plt.title('Problem 2B Training Loss')
plt.savefig("figures/Problem 2B Training Loss.png", facecolor='w', dpi=300)