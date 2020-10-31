import torch
import torch.nn as nn
import torchvision.models as models

'''
EncoderCNN class provided by Udacity (utilizing Resnet)
was used as is by me, while attempting this project.
'''
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
                
        '''
        [See the diagram of the decoder in Notebook 1]
        The RNN needs to have 4 basic components :
        
        1. Word Embedding layer : maps the captions to embedded word vector of embed_size.
        2. LSTM layer : inputs( embedded feature vector from CNN , embedded word vector ).
        3. Hidden layer : Takes LSTM output as input and maps it 
                          to (batch_size, caption length, hidden_size) tensor.
        4. Linear layer : Maps the hidden layer output to the number of words
                          we want as output, vocab_size.
                          
        NOTE : I did not define any init_hidden method based on the discussion 
               in the following thread in student hub.
               Hidden state defaults to zero when nothing is specified, 
               thus not requiring the need to explicitly define init_hidden.
               
        [https://study-hall.udacity.com/rooms/community:nd891:682337-project-461/community:thread-11927138595-435532?contextType=room]
                          
        '''
        
        super().__init__()

        '''
         vocab_size : size of the dictionary of embeddings, 
                      basically the number of tokens in the vocabulary(word2idx) 
                      for that batch of data.
         embed_size : the size of each embedding vector of captions
        '''
        self.word_embedding_layer = nn.Embedding(vocab_size, embed_size)
        
        
        '''
        LSTM layer parameters :
        
        input_size  = embed_size 
        hidden_size = hidden_size     # number of units in hidden layer of LSTM  
        num_layers  = 1               # number of LSTM layers ( = 1, by default )
        batch_first = True            # input , output need to have batch size as 1st dimension
        dropout     = 0               # did not use dropout 
        
        Other parameters were not changed from default values provided in the PyTorch implementation.
        '''
        self.lstm = nn.LSTM( input_size = embed_size, 
                             hidden_size = hidden_size, 
                             num_layers = num_layers, 
                             dropout = 0, 
                             batch_first=True
                           )
        
        self.linear_fc = nn.Linear(hidden_size, vocab_size)

    
    def forward(self, features, captions):
        
        '''
        
        Arguments :
        
        For a forward pass, the instantiation of the RNNDecoder class
        receives as inputs 2 arguments  :
        
        -> features : ouput of CNNEncoder having shape (batch_size, embed_size).
        -> captions : a PyTorch tensor corresponding to the last batch of captions 
                      having shape (batch_size, caption_length) .
        
        NOTE : Input parameters have first dimension as batch_size.
        
        '''
        
        # Discard the <end> word to avoid the following error in Notebook 1 : Step 4
        # (outputs.shape[1]==captions.shape[1]) condition won't be satisfied otherwise.
        # AssertionError: The shape of the decoder output is incorrect.
        captions = captions[:, :-1] 
        
        # Pass image captions through the word_embeddings layer.
        # output shape : (batch_size, caption length , embed_size)
        captions = self.word_embedding_layer(captions)
        
        # Concatenate the feature vectors for image and captions.
        # Features shape : (batch_size, embed_size)
        # Word embeddings shape : (batch_size, caption length , embed_size)
        # output shape : (batch_size, caption length, embed_size)
        inputs = torch.cat((features.unsqueeze(1), captions), dim=1)
        
        # Get the output and hidden state by passing the lstm over our word embeddings
        # the hidden state is not used, so the returned value is denoted by _.
        # Input to LSTM : concatenated tensor(features, embeddings) and hidden state
        # output shape : (batch_size, caption length, hidden_size)
        outputs, _ = self.lstm(inputs)
        
        # output shape : (batch_size, caption length, vocab_size)
        # NOTE : First dimension of output shape is batch_size.
        outputs = self.linear_fc(outputs)
        
        return outputs

    
    def sample(self, inputs, states=None, max_len=20):
        '''
          Arguments : accepts pre-processed image tensor (inputs) 
          Returns   : predicted sentence (list of tensor ids of length max_len)
          
          Implementation details :
          
          features : (batch_size , embed_size) [ See Notebook 1 ]
          inputs = features.unsqueeze(1) : (batch_size , 1, embed_size) [ See Notebook 3 ]
          sample function is used for only 1 image at a time. Thus, batch_size = 1
          input shape : (1,1,embed_size)
          
          shape of LSTM output : (batch_size,caption length, hidden_size)
          The input has to be fed to the lstm till the <end> is reached.
          every time the input is fed to the lstm, caption of length 1 is produced by the RNNDecoder.
          Thus LSTM output shape : (1,1,hidden_size)

          LSTM output is linear_fc input. (This is wrong.  See NOTE )
          shape of input : (1,1,hidden_size)
          shape of output : (1,1,vocab_size)
          
          NOTE :
          
          Even after training my model, I was getting as output a sequence of <unk> in [ Notebook 3 ].
          So, I looked at answers in Knowledge and discussions in Student Hub.
          
          The following thread in the student hub gave me intuition for :
          
          1. Passing states as an input to the LSTM layer in the sample() function.
          2. Squeezing the ouput of LSTM layer before passing it to the Linear layer.
          
          https://study-hall.udacity.com/rooms/community:nd891:682337-project-461/community:thread-11730509939-428812?contextType=room

        '''
        # The output of this function will be a Python list of integers,
        # indicating the corresponding token words in the dictionary.
        outputs = []   
        output_length = 0
        
        while (output_length != max_len+1):
            
            ''' LSTM layer '''
            # input  : (1,1,embed_size)
            # output : (1,1,hidden_size)
            # States should be passed to LSTM on each iteration in order for it to recall the last word it produced.
            output, states = self.lstm(inputs,states)
           
            ''' Linear layer '''
            # input  : (1,hidden_size)
            # output : (1,vocab_size)
            output = self.linear_fc(output.squeeze(dim = 1))
            _, predicted_index = torch.max(output, 1)
            
            # CUDA tensor has to be first converted to cpu and then to numpy.
            # Because numpy doesn't support CUDA ( GPU memory ) directly.
            # See this link for reference : https://discuss.pytorch.org/t/convert-to-numpy-cuda-variable/499
            outputs.append(predicted_index.cpu().numpy()[0].item())
            
            # <end> has index_value = 1 in the vocabulary [ Notebook 1 ]
            # This conditional statement helps to break out of the while loop,
            # as soon as the first <end> is encountered. Length of caption maybe less than 20 at this point.
            if (predicted_index == 1):
                break
            
            # Prepare for net loop iteration 
            # Embed the last predicted word to be the new input of the LSTM
            # To understand this step, again look at the diagram at end of  [ Notebook 1 ]
            inputs = self.word_embedding_layer(predicted_index)   
            inputs = inputs.unsqueeze(1)
            
            # To move to the next iteration of the while loop.
            output_length += 1

        return outputs
        
         