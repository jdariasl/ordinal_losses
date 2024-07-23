import torch
import torch.nn.functional as F

def post_processing_prob_ordinal(model_output):

  epsilon = torch.tensor(1e-6, dtype=torch.float32)
  theta = model_output[:,0:-1]
  batch_size = theta.shape[0]
  input = model_output[:,-1].reshape(batch_size,1)
  theta_values = torch.cumsum(torch.clamp(F.softplus(theta), epsilon, torch.tensor(1e20)),1)
  sigmoid_est_mean = F.sigmoid(theta_values - input)
  mean_probs = torch.cat([sigmoid_est_mean, torch.ones((batch_size, 1), dtype=torch.float32)], dim=1) - \
       torch.cat([torch.zeros((batch_size, 1), dtype=torch.float32), sigmoid_est_mean], dim=1)
  mean_probs = torch.clamp(mean_probs,epsilon,torch.tensor(1.0))
  predicted_label = torch.argmax(mean_probs, axis=-1)
  return mean_probs, predicted_label
