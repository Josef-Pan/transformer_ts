# Probability Time Series Transformer
Training Time Series Transfomer from Scratch based on Huggingface Transformer
![Screenshot 2023-10-16 at 20 02 39](https://github.com/Josef-Pan/transformer_ts/assets/20598795/c978eaa4-d69a-4771-9574-4497a233c516)
![Screenshot 2023-10-16 at 20 03 09](https://github.com/Josef-Pan/transformer_ts/assets/20598795/6eb18328-12d1-4924-b747-0f5faa028d95)
We can see from the above results that after 130 epochs, the prediction is quite accurate.
## Datasource using csv file and imported as pandas dataframe
## Gluonts framework was not used to make the training process transparent
## Many parameters changed for better performance
- d_model changed to 128 from default 32
- encoder_attention_heads changed to 8
- encoder_layers changed to 16
- decoder_layers changed to 32
- decoder_layers is more important in this case, but you need to consider the GPU memory.
