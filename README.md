# Domain-Adaptation
Read the complete report [here](/Report.pdf)

A conceptual analysis to use domain adaptation for classification of documents as "Sports" "Politics" and "Technology"

To run the program please follow the below steps.
	
	1. Download 20Newsgroup dataset @ http://qwone.com/~jason/20Newsgroups/20news-19997.tar.gz
	
	2. Extract the data set and copy the folder "20_Newsgroup" to this location
	
	3. Download BBC News dataset @ http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip
	
	4. Extract the data set and copy the folder "bbc_full" to this location
	
	5. Please make sure the following packages are installed.
		a. Python 2.7.3 or higher
		b. numpy - pip install numpy
		c. scipy - pip install scipy
		d. scikit learn - pip install scikit-learn
	
	6. We need some preprocessing for the data first, to do this open a terminal pointing to this location and enter the following command.
		make clean-input
	
	7. Now you can run the code.
	   (The program will take a few seconds to load data and start), 
	   Command for LogisticRegression with Stochastic Gradient Descent:
		make domain-adaptation-max-ent
			OR
	   Command for Multinomial Naive Bayes:
		make domain-adaptation-naive-bayes
