import sys
sys.path.append("./utils/coco/pycocoevalcap/")


from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu

class COCOEvalCap:
    def __init__(self, cocoRes, captions, dataset):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
# 	self.coco = coco
        self.cocoRes = cocoRes
        self.captions = captions
        self.dataset = dataset
        self.params = {'image_id': dataset.image_ids}

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for i in range(len(self.captions)):
          s = self.captions.iloc[i, 0].split('.')[0]
          if s in gts.keys():
            gts[s].append(self.captions.iloc[i, 1])
          else:
            gts[s] = [self.captions.iloc[i, 1]] 
            res[s] = self.cocoRes.imgToAnns[s] 
#        i = 5
#        c = 0
#        for imgId in imgIds:
#          anns = []
#          while c < i and c < len(self.captions):
#            anns.append(self.captions.iloc[c, 1])
#            c = c + 1
#          i = i + 5
#          gts[imgId] = anns
#          res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])
            # (Meteor(),"METEOR"),
            # (Rouge(), "ROUGE_L"),
            # (Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
