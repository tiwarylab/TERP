"""
SPIB: A deep learning-based framework to learn RCs 
from MD trajectories. Code maintained by Dedi.

Read and cite the following when using this method:
https://aip.scitation.org/doi/abs/10.1063/5.0038198
"""
import numpy as np
import torch
import os
import sys

import SPIB
import SPIB_training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
default_device = torch.device("cpu")


def test_model():
    # Settings
    # ------------------------------------------------------------------------------
    # By default, we save all the results in subdirectories of the following path.
    base_path = "SPIB"
    
    # Model parameters
    # Time delay delta t in terms of # of minimal time resolution of the trajectory data
    if '-dt' in sys.argv:
        dt = int(sys.argv[sys.argv.index('-dt') + 1])
    else:
        dt = 10
    
    # By default, we use all the all the data to train and test our model
    t0 = 0 
    
    # Dimension of RC or bottleneck
    if '-d' in sys.argv:
        RC_dim = int(sys.argv[sys.argv.index('-d') + 1])
    else:
        RC_dim = 2
    
    # Encoder type ('Linear' or 'Nonlinear')
    if '-encoder_type' in sys.argv and (sys.argv[sys.argv.index('-encoder_type') + 1])=='Nonlinear':
        encoder_type = 'Nonlinear'
    else:
        encoder_type = 'Linear'

    # Number of nodes in each hidden layer of the encoder
    if '-n1' in sys.argv:
        neuron_num1 = int(sys.argv[sys.argv.index('-n1') + 1])
    else:
        neuron_num1 = 16
    # Number of nodes in each hidden layer of the encoder
    if '-n2' in sys.argv:
        neuron_num2 = int(sys.argv[sys.argv.index('-n2') + 1])
    else:
        neuron_num2 = 16
    
    
    # Training parameters
    
    if '-bs' in sys.argv:
        batch_size = int(sys.argv[sys.argv.index('-bs') + 1])
    else:
        batch_size = 2048

    # Threshold in terms of the change of the predicted state population for measuring the convergence of the training
    if '-threshold' in sys.argv:
        threshold = float(sys.argv[sys.argv.index('-threshold') + 1])
    else:
        threshold = 0.01

    # Number of epochs with the change of the state population smaller than the threshold after which this iteration of the training finishes
    if '-patience' in sys.argv:
        patience = int(sys.argv[sys.argv.index('-patience') + 1])
    else:
        patience = 0

    # Minimum refinements
    if '-min_refinements' in sys.argv:
        min_refinements = int(sys.argv[sys.argv.index('-min_refinements') + 1])
    else:
        min_refinements = 0
        
    # By default, we save the model every 10000 steps
    log_interval = 10000 
    
    # By default, there is no learning rate decay
    lr_scheduler_step_size = 1
    lr_scheduler_gamma = 1

    # Initial learning rate of Adam optimizer
    if '-lr' in sys.argv:
        learning_rate = float(sys.argv[sys.argv.index('-lr') + 1])
    else:
        learning_rate = 1e-3
    
    # Hyper-parameter beta
    if '-b' in sys.argv:
        beta = float(sys.argv[sys.argv.index('-b') + 1])
    else:
        beta = 1e-3
    
    # Import data
    
    # Path to the initial state labels
    if '-label' in sys.argv:
        initial_label = np.load(sys.argv[sys.argv.index('-label') + 1])
    else:
        print("Pleast input the initial state labels!")
        return
    
    traj_labels = torch.from_numpy(initial_label).float().to(default_device)
    output_dim = initial_label.shape[1]
    
    # Path to the trajectory data
    if '-traj' in sys.argv:
        traj_data = np.load(sys.argv[sys.argv.index('-traj') + 1])
    else:
        print("Pleast input the trajectory data!")
        return
    
    traj_data = torch.from_numpy(traj_data).float().to(default_device)
    
    
    # Path to the weights of the samples
    if '-w' in sys.argv:
        traj_weights = np.load(sys.argv[sys.argv.index('-w') + 1])
        traj_weights = torch.from_numpy(traj_weights).float().to(default_device)
        IB_path = os.path.join(base_path, "Weighted")
    else:
        traj_weights = None
        IB_path = os.path.join(base_path, "Unweighted")
    
    # Random seed
    if '-seed' in sys.argv:
        seed = int(sys.argv[sys.argv.index('-seed') + 1])
        np.random.seed(seed)
        torch.manual_seed(seed)    
    else:
        seed = 0
    
    
    # Other controls
    
    # Whether to refine the labels during the training process
    if '-UpdateLabel' in sys.argv:
        UpdateLabel = bool(sys.argv[sys.argv.index('-UpdateLabel') + 1])  
    else:
        UpdateLabel = True
    
    
    # Whether save trajectory results
    if '-SaveTrajResults' in sys.argv:
        SaveTrajResults = bool(sys.argv[sys.argv.index('-SaveTrajResults') + 1])  
    else:
        SaveTrajResults = True
    
    # Train and Test our model
    # ------------------------------------------------------------------------------
    
    final_result_path = IB_path + '_result.dat'
    os.makedirs(os.path.dirname(final_result_path), exist_ok=True)
    print("Final Result", file=open(final_result_path, 'w'))
    
    data_shape, train_past_data, train_future_data, train_data_labels, train_data_weights, \
        test_past_data, test_future_data, test_data_labels, test_data_weights = \
            SPIB_training.data_init(t0, dt, traj_data, traj_labels, traj_weights)
    
    output_path = IB_path + "_d=%d_t=%d_b=%.4f_learn=%f" \
        % (RC_dim, dt, beta, learning_rate)

    IB = SPIB.SPIB(encoder_type, RC_dim, output_dim, data_shape, device, \
                   UpdateLabel, neuron_num1, neuron_num2)
    
    IB.to(device)
    
    # use the training set to initialize the pseudo-inputs
    IB.init_representative_inputs(train_past_data, train_data_labels)

    train_result = False
    
    optimizer = torch.optim.Adam(IB.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)

    train_result = SPIB_training.train(IB, beta, train_past_data, train_future_data, \
                                       train_data_labels, train_data_weights, test_past_data, test_future_data, \
                                           test_data_labels, test_data_weights, optimizer, scheduler,\
                                               batch_size, threshold, patience, min_refinements, output_path, \
                                                   log_interval, device, seed)
    
    if train_result:
        return
    
    SPIB_training.output_final_result(IB, device, train_past_data, train_future_data, train_data_labels, train_data_weights, \
                                      test_past_data, test_future_data, test_data_labels, test_data_weights, batch_size, \
                                          output_path, final_result_path, dt, beta, learning_rate, seed)

    IB.save_traj_results(traj_data, batch_size, output_path, SaveTrajResults, 0, seed)
    
    IB.save_representative_parameters(output_path, seed)


if __name__ == '__main__':
    
    test_model()
    

    
    
