PRE-RUN CHECKLIST:

make sure you have all the following files are in the same folder:
- main.py
- utils.py
- pca.py
- dataExtractor3000.py
- hierarchicalClustering.py
- classification.py
- offensive-data-2023-24.csv
if you wont have offensive-data-2023-24.csv, the program will automatically try and get the data from the API, which might take a long time (over an hour in some instances)


PROGRAM RUNNING INSTRUCTIONS:

Install Dependencies:

It is recommended to create an seperate python environment and install this dependencies in that environment.
Command to create a python environment:
python -m venv nba
Activate Environment: 
nba\Scripts\activate

We have a text file "requirements.txt" for dependency management. To install all the dependencies, use following command: 
pip install -r requirements.txt


run main.py , you will get 2 pictures pop ups (dendograms if you use hierarchical clustering, elbow functions if you use kMean), first you will get a picture explaining the offensive clusters, then a picture explaining the defensive clusters. you need to close those pictures before the program keeps running, not minimize, but actually close them. all the output will be in the terminal.

PROGRAM MODIFICATIONS:

if you want to play with different clustering methods, in main.py you can choose what type of method you wan to use in the variable offenseMethod and defenseMethod. the clustering options are: 'pca_kmeans', 'pca_hierarchical', 'kmeans', 'hierarchical'. 
the variables offenseClusters and defenseClusters in main.py are to set the amount of clusters you want if you use kmeans or pca_kmeans
the variables offenseDistance and defenseDistance in main.py are to set the distance between clusters if you use hierarchical or pca_hierarchical
the variables offensePcaComponents and defensePcaComponents in main.py are to set the number of PCA components you want the program to use if you use pca_kmeans or pca_hierarchical
you can change the variable seasonData in utils.py to change the season but WARNING: this will cause the program to get the data from the API which will take a long time.
