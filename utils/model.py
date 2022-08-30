import torch
from transformers import LayoutLMModel, LayoutLMConfig
from torchvision import models

# Conv1d layers after the resnet/layoutlm
class Conv1dBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, p_dropout=0.2):
        super().__init__()
        self.p_dropout = p_dropout
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same', padding_mode='circular')
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=self.p_dropout)
    
    def forward(self, x):
        x = self.conv1d(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

class Conv1dLayers(torch.nn.Module):
    def __init__(self, kernel_size, num_hidden_features):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_hidden_features = num_hidden_features
        
        self.num_features_0=768+512
            
        self.num_features = [self.num_features_0] + self.num_hidden_features #768+512
        
        if len(self.num_features)>1:
            self.stack_conv1d = torch.nn.ModuleList([
                Conv1dBlock(in_channels=self.num_features[n-1], out_channels=self.num_features[n], kernel_size=kernel_size)
                for n in range(1, len(self.num_features))])
        
        self.conv1d = torch.nn.Conv1d(in_channels=self.num_features[-1], out_channels=2, kernel_size=kernel_size, padding='same', padding_mode='circular')
#       no self.activation = torch.nn.Softmax(dim=1) because softmax is combined in CrossEntropyLoss  

    def forward(self, *embeddings): # ocr: (merge_size, 768), img: (merge_size, 512) 
        x = torch.cat(embeddings,dim=1) 
        x = x.permute(1, 0) # (merge_size, 768+512) -> (768+512, merge_size)
        x = torch.unsqueeze(x, 0) # (768+512, merge_size) -> (1, 768+512, merge_size)
        if len(self.num_features)>1:
            for layer in self.stack_conv1d:
                x = layer(x) # (1, 768+512, merge_size) ->->-> 
        x = self.conv1d(x) # -> (1, 2, merge_size)
        x = x.permute(0, 2, 1) # (1, w, merge_size) -> (1, merge_size, 2)
        x = torch.squeeze(x, dim=0) # (1, merge_size, 2) -> (merge_size, 2)
#       no x = self.activation(x) because softmax is combined in CrossEntropyLoss
        return x #(merge_size, 2)

# Cut out the last fc layer (512->1000-dim vector) in resnet
class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class ModifiedResNet(torch.nn.Module):
    def __init__(self, pretrained_weights=True):
        super().__init__()
        self.pretrained_weights = pretrained_weights
        self.resnet = models.resnet18(pretrained=pretrained_weights)
        self.resnet.fc = Identity() # cut out the last layer

    def forward(self, x):
        return self.resnet(x) # output = (batch, 512)

class PDFSegmentationModel(torch.nn.Module):
    def __init__(self, num_hidden_features, kernel_size=3, pretrained_weights=True, freeze=False):
        super().__init__()
        '''
            Args:
        '''
        self.kernel_size = kernel_size
        self.num_hidden_features = num_hidden_features
        self.pretrained_weights = pretrained_weights
        self.freeze = freeze
        
        self.validation_loss = None
        self.validation_f1 = None


        self.layoutlm = LayoutLMModel.from_pretrained('microsoft/layoutlm-base-uncased') # (merge_size, ...) -> (merge_size, 768)
        self.resnet = ModifiedResNet(pretrained_weights=pretrained_weights)
        
        if freeze: # freeze the layoutlm and resnet part in the ablation
            for param in self.layoutlm.parameters():
                param.requires_grad = False
            for param in self.resnet.parameters():
                param.requires_grad = False
    
        self.conv1dlayers = Conv1dLayers(kernel_size=kernel_size, num_hidden_features=num_hidden_features)
    
    def forward(self, input_ids, bbox, attention_mask, token_type_ids, image):
        img_embeddings = self.resnet(image)
        ocr_embeddings = self.layoutlm(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)['pooler_output']
        
        outputs = self.conv1dlayers(ocr_embeddings, img_embeddings)
        return outputs