#VideoUnderstanding
 
My attempt at making computers "understand" videos. I started my code with Andrej Karpathy's Image Captioning model NeuralTalk [NeuralTalk](https://github.com/karpathy/neuraltalk) but ended up doing something else due to open-world captioning performance issues.  

This was my Computer Vision undergraduate thesis applying Faster RCNN+YOLO+Scene Classification+Scene splitting+scene change detection with results all displayed in fancy Flask web app with sockets and much more. Kenneth Dawson-Howe was my supervisor. Images below showing the web app:

Landing Page showing previously chosen and processed YouTube videos with "TV room" effect. Any YouTube video could be selected but collage videos provided the best results for scene change detection.
![landing page](docs/images/landingpage1.png)  

Console showing print commands in the underlying Flask server sent over a sockets interface 
![console1](docs/images/console1.png)  
![console3](docs/images/console3.png)  

Actual results page for a specific video. Horizontal scroll and general information about video
![results1](docs/images/results1.png)  

Showing scene results aggregated from frames within scene. 
![results2](docs/images/results2.png)  

Faster R-CNN and YOLO were used to detect specific objects in each frame
![results4](docs/images/results4.png)  

K-means, average colour were computed for each frame and chi-distance between consecutive frame histograms was computed to calculate scene change detection
![results5](docs/images/results5.png)  
![results6](docs/images/results6.png)  
