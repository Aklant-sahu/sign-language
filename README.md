## This Project is aimed at Translating Sign Language to Human Readable Characters using Machine Learning
The Pipeline of the project includes-    
1) Creating our own dataset using Opencv and MediaPipe Library  
2) Data points or the feature columns are the index locations of all the main points of a human hand.  
3)The Datset is captured for every character from A-Z ,Joined and the distributed randomly to form a better distribution for model training  
4) Do Feature selection ,Data cleaning and that apply SVM model  
5) Train the modle using SVM  
6) Test the model.Accuracy is around -98%  

Main Advantages of the model include-  
Since the model is Created not on the images but rather using the coordinates of the index locations of the hand,it is resistant to different background and image conditions.Hence this Model generalizes a lot better than the Previous existing models.The key for this project is not the architecture of the modle but rather how the data was collected and used to solve the problem of generalization.  

![image](https://user-images.githubusercontent.com/88976862/192091639-55ca620c-ff33-433a-bb93-d1fd1a1ec7a0.png)
![image](https://user-images.githubusercontent.com/88976862/192091645-e84cf55d-c793-4821-b7b1-2f682b1f7d63.png)



