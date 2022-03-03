# AI-Server
An HTTP server who responded to post requests with the result of running a pre-trained model on a provided image.

## Notes:

(I built this circa August 2021)  
At the time I was testing the feasibility of executing machine learning models on advanced hardware by querying them over the internet. To do so I developed this server who responded to properly formatted requests with the output of a specific pre-trained model. In this case, the test was digit recognition.  


ai_test.py was used to produce the model that the server would later load.  
data.py was used to produce img0.jpg from the img.npz file.
