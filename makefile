clean-input:
	cd bbc_full;rm -rf business;rm -rf entertainment;cd ../20_newsgroups;rm -rf alt.atheism;rm -rf misc.forsale;rm -rf rec.autos;rm -rf rec.motorcycles;rm -rf soc.religion.christian;rm -rf talk.religion.misc;mkdir politics;mkdir sport;mkdir tech;mv comp.*/* tech;mv sci.*/* tech;mv talk.politics.*/* politics;mv rec.sport.*/* sport;rm -rf comp.*;rm -rf sci.*;rm -rf rec.*;rm -rf talk.*;

domain-adaptation-max-ent:
	python domainAdaptation.py LogisticRegression

domain-adaptation-naive-bayes:
	python domainAdaptation.py MultinomialNB
