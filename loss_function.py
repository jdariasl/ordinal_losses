
import torch
import torch.nn.functional as F
#See: Arias-Londoño, J. D., Gómez-García, J. A., & Godino-Llorente, J. I. (2019). 
#Multimodal and multi-output deep learning architectures for the automatic assessment 
#of voice quality using the GRB scale. IEEE Journal of Selected Topics in Signal Processing, 14(2), 413-422.

#Weigthed ordinal cross-entropy loss
class ordinal_ce_loss(nn.Module):
    def __init__(self, class_weigth=None):
        super(ordinal_loss, self).__init__()
        if class_weigth is None:
            self.CE_loss = torch.nn.CrossEntropyLoss(reduce=False)
        else:
            self.CE_loss = torch.nn.CrossEntropyLoss(weight=class_weigth, reduce=False)

    def forward(self, input, target):
        j = torch.argmax(input, dim=1)
        v = 1 + torch.abs(j - target)
        loss = self.CE_loss(input, target)
        return torch.mean(v * loss)


#Weigthed ordinal binary cross-entropy
class ordinal_binary_loss(nn.Module):
    def __init__(self, class_weigth=None):
        super(ordinal_loss, self).__init__()
        if class_weigth is None:
            self.Bi_loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.Bi_loss = torch.nn.BCEWithLogitsLoss(weight=class_weigth)

    def thermometer_encoding(self,values, num_classes):
        """
        Convert a list of ordinal values into thermometer encoded representations.
        
        Parameters:
        values (torch.Tensor): Tensor of ordinal values (integers starting from 1).
        num_classes (int): The number of classes or the maximum value in the scale.
        
        Returns:
        torch.Tensor: A tensor containing the thermometer encoded representations.
        
        Example:
        For values = [1, 2, 3, 4, 5] and num_classes = 5,
        the thermometer encoding would be:
        [[1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]]
        """
        batch_size = values.shape[0]
        thermometer_encoded = torch.zeros(batch_size, num_classes, dtype=torch.float32)
        
        for i in range(num_classes):
            thermometer_encoded[:, i] = (values > i).float()
        
        return thermometer_encoded

    def forward(self, input, target):

        #This loss function requires the target to use a thermometer enconding
        #so we must convert it
        num_classes = input.shape[-1]
        target = self.thermometer_encoding(target, num_classes)
        loss = self.Bi_loss(input, target)
        return loss


#Probabilistic ordinal loss
#See: Nazabal, A., Olmos, P. M., Ghahramani, Z., & Valera, I. (2020). Handling 
#incomplete heterogeneous data using vaes. Pattern Recognition, 107, 107501.
class probabilistic_ordinal_loss(nn.Module):
    def __init__(self, class_weigth=None):
        super(ordinal_loss, self).__init__()
        epsilon = torch.tensor(1e-6, dtype=tf.float32)
        if class_weigth is None:
            self.CE_loss = torch.nn.CrossEntropyLoss(reduce=False)
        else:
            self.CE_loss = torch.nn.CrossEntropyLoss(weight=class_weigth, reduce=False)

    def forward(self, theta, input, target):
        #theta are the logits predicted thresholds [batch,n_classes-1]
        #input is the predicted regression value (one-dimensional) [batch,]
        #target must a tensor with int numbers indicating the target class [batch,]
        batch_size = theta.shape[0]
        input = input.reshape(-1,1)
        theta_values = torch.cumsum(torch.clamp(F.softplus(theta), self.epsilon, torch.tensor(1e20)),1)
        sigmoid_est_mean = F.sigmoid(theta_values - input)
        mean_probs = torch.cat([sigmoid_est_mean, torch.ones((batch_size, 1), dtype=torch.float32)], dim=1) - \
             torch.cat([torch.zeros((batch_size, 1), dtype=torch.float32), sigmoid_est_mean], dim=1)
        mean_probs = torch.clamp(mean_probs,epsilon,torch.tensor(1.0))
        log_probs = torch.log(mean_probs)
        return -self.CE_loss(log_probs, target)