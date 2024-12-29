

class Score:
    """
    a score class for metric
    """
    def __init__(self,name, val,upper=False):
        """
        name: name of this metric  e.g. fid,  ssim
        :param val: init value for the metric
        :param upper: if the upper ==True the higher is better, otherwise the lower is better
        """
        self.name = name;
        self.best_val = val;
        self.upper = upper;
    def update(self,val):
        if (self.compare(val)):
            self.best_val = val

    def compare(self,val):
        # the higher is the better
        if (val>self.best_val and self.upper==True):
            return True
        #the lower if the better
        elif(val<self.best_val and self.upper==False):
            return True
        return False

    def get_best(self):
        return self.best_val
    def get_name(self):
        return self.name
    def get_type(self):
        return self.upper

class ScoreManager:
    """
    The calss for manage the Scores
    """
    def __init__(self,kwargs ={}):
        """
        :param kwargs:  for this format {"fid":[9999, -1]}
        999 indicate the init value, -1 indicates that the lower is ther better

        """
        self.score_dic = {}
        for key in kwargs:
            name = key
            init_val = kwargs[key][0]
            upper = kwargs[key][1]>0
            self.score_dic[name] = Score(name=name,val=init_val,upper=upper)

    def update(self,kwargs ={}):
        """
        :param kwargs:
        :return:  for this format {"fid":6.6}
        "fid" indicates the name of the metric, 6.6 corresponds the update value
        """
        for key in kwargs:
            name = key
            score = kwargs[key]
            if name in self.score_dic:
                self.score_dic[name].update(score)
            # else:
            #     self.score_dic[name] = Score(name=name, init_val=init_val, upper=upper)

    def get_all_dic(self):
        out_dic = {}
        for key in self.score_dic:
            score = self.score_dic[key]
            upper_score = -1
            val = score.get_best()
            if score.get_type()==True:
                upper_score = 1
            out_dic[score.get_name()] = [val,upper_score]
        return out_dic

    def compare(self,key,kwargs={}):
        sorce = self.score_dic[key]
        if sorce.compare(kwargs[key]):
            return True
        return False


    def get_metric(self,name):
        return self.score_dic[name].get_best()




