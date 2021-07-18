from dataset import np
from dataset import df1,df2,df3,df4,df5,df6,df7,df,Folds
class CreateTestandTrain:
    def __init__(self):
        self.dict={}
        self.reverse_dict={}

    def create_dict(self):
        indices=0
        for index, row in df.iterrows():
            temp=row['data'].split(' ')
            for i in range(0,len(temp)):
                if temp[i] in self.dict.keys():
                    continue
                self.dict[temp[i]]=indices
                indices+=1

    def create_revdict(self):

        for i in self.dict:
            self.reverse_dict[self.dict[i]]=i


    def create_train_data(self):
        self.train1=self.append_dataset(df1,df2,df3,df4,df5,df6)
        self.train2 = self.append_dataset(df7, df2, df3, df4, df5, df6)
        self.train3 = self.append_dataset(df1, df7, df3, df4, df5, df6)
        self.train4 = self.append_dataset(df1, df2, df7, df4, df5, df6)
        self.train5 = self.append_dataset(df1, df2, df3, df7, df5, df6)
        self.train6 = self.append_dataset(df1, df2, df3, df4, df7, df6)
        self.train7 = self.append_dataset(df1, df2, df3, df4, df5, df7)
        self.test1=df7
        self.test2 = df1
        self.test3 = df2
        self.test4 = df3
        self.test5 = df4
        self.test6 = df5
        self.test7 = df6



    def append_dataset(self,df1,df2,df3,df4,df5,df6):
        return df1.append(df2.append(df3.append(df4.append(df5.append(df6)))))


class PreProcess:
    def __init__(self):
        pass
    def tf_vectors(self,train,dataset):
        self.spam=0
        self.non_spam=0
        self.tf=np.zeros([len(dataset.dict),3])

        for index,row in train.iterrows():
            temp=row['data'].split(' ')

            if int(row['value'])==0:
                self.spam+=1
            else:
                self.non_spam+=1

            for i in temp:
                self.tf[dataset.dict[i]][int(row['value'])]+=1
        self.spam_prob=self.spam/(self.spam+self.non_spam)


class ClassificationUsingNaiveBayes:
    def __init__(self):
        self.efficency=np.zeros([Folds,3])
    def testing_data(self,train_data_tf,spam_prob,test_data,dictionary,spam,ham,fold):
        rightly_classify=0
        wrongly_classify=0
        doc_size=len(dictionary)
        for index, row in test_data.iterrows():
            temp=row['data'].split(' ')
            prob_spam=spam_prob
            prob_not_spam=1-spam_prob
            for i in temp:
                prob_spam*=self.naive_bayes(train_data_tf[dictionary[i]][0],doc_size,spam)
                prob_not_spam*=self.naive_bayes(train_data_tf[dictionary[i]][1],doc_size,ham)

            if (prob_spam>prob_not_spam and int(row['value'])==0) or(prob_spam<prob_not_spam and int(row['value'])==1) :
                rightly_classify+=1
            else:
                wrongly_classify+=1



        efficency=rightly_classify/(rightly_classify+wrongly_classify)
        self.efficency[fold-1][0] = rightly_classify
        self.efficency[fold-1][1]= wrongly_classify
        self.efficency[fold-1][2]= efficency


        print("*********Results of fold "+str(fold)+" is*********")
        print("Rightly Classified Mails are "+ str(rightly_classify)+" Wrongly classified Mails are "+ str(wrongly_classify)+"  and efficency is "+str('%.2f'%(efficency*100))+'%')
        print()

    def print_final_answer(self):
        print("*********Results of overall 7 Fold Cross Validation is*********")
        print("Rightly Classified Mails are "+ str(np.sum(self.efficency.T[0]))+" Wrongly classified Mails are "+ str(np.sum(self.efficency.T[1]))+"  and efficency is "+str('%.2f'%((np.sum(self.efficency.T[2])/Folds)*100))+'%')

    def naive_bayes(self,a,b,c):
        return (a+1)/(b+c)







c=CreateTestandTrain()
c.create_dict()
c.create_revdict()
c.create_train_data()

p=PreProcess()

naive=ClassificationUsingNaiveBayes()

p.tf_vectors(c.train1,c)
naive.testing_data(p.tf,p.spam_prob,c.test1,c.dict,p.spam,p.non_spam,1)

p.tf_vectors(c.train2,c)
naive.testing_data(p.tf,p.spam_prob,c.test2,c.dict,p.spam,p.non_spam,2)

p.tf_vectors(c.train3,c)
naive.testing_data(p.tf,p.spam_prob,c.test3,c.dict,p.spam,p.non_spam,3)

p.tf_vectors(c.train4,c)
naive.testing_data(p.tf,p.spam_prob,c.test4,c.dict,p.spam,p.non_spam,4)

p.tf_vectors(c.train5,c)
naive.testing_data(p.tf,p.spam_prob,c.test5,c.dict,p.spam,p.non_spam,5)

p.tf_vectors(c.train6,c)
naive.testing_data(p.tf,p.spam_prob,c.test6,c.dict,p.spam,p.non_spam,6)

p.tf_vectors(c.train7,c)
naive.testing_data(p.tf,p.spam_prob,c.test7,c.dict,p.spam,p.non_spam,7)

naive.print_final_answer()