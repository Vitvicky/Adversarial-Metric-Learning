import torch
import numpy as np
from networks import EmbeddingNet, TripletNet
from losses import TripletLoss
from network import *
from loss import *

def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)  

def train_epoch(train_loader, model, cuda, margin, lambda1, lambda2):
	# Model parameters
    g_input_size = 1      # Random noise dimension coming into generator, per output vector
    g_hidden_size = 50     # Generator complexity
    g_output_size = 100     # Size of generated output vector
    d_input_size = 784    # Minibatch size - cardinality of distributions
    d_hidden_size = 300    # Discriminator complexity
    d_output_size = 100     # Single dimension for 'real' vs. 'fake' classification
	optim_betas = (0.9, 0.999)
	
	gi_sampler = get_generator_input_sampler()
	
    model.train()
    losses = []
    total_loss = 0
	
	G = Generator(input_size=g_input_size,
                  hidden_size=g_hidden_size,
                  output_size=g_output_size,
                  f=generator_activation_function)
	g_optimizer = optim.Adam(G.parameters(), lr=0.01, betas=optim_betas)
	net_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
	
	criterion_adv = generateLoss(margin, lambda1, lambda2)  
	criterion_triplet = TripletLoss(margin)
	
	# start to train
    for batch_idx, (data1, data2, data3) in enumerate(train_loader):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)


        net_optimizer.zero_grad()
		g_optimizer.zero_grad()
		
        embedded_x, embedded_y, embedded_z = model(data1, data2, data3)

		# generate adversarial example
		adv_example = ()
		
        loss_adv = criterion_adv(embedded_x, adv_example, embedded_y, embedded_z)
		loss_trip = criterion_triplet(embedded_x, embedded_y, embedded_z)
        loss = loss_adv + 0.001 * loss_trip
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        net_optimizer.step()

       
        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics

	
margin = 1.
embedding_net = EmbeddingNet()
model = TripletNet(embedding_net)
if cuda:
    model.cuda()