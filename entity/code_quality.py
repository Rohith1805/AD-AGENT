class CodeQuality:
    def __init__(self,code,algorithm,parameters,std_output,error_message,auroc,auprc,error_points,review_count):
        self.code = code
        self.algorithm = algorithm
        self.parameters = parameters
        self.std_output = std_output
        self.error_message = error_message
        self.auroc = auroc
        self.auprc = auprc
        self.error_points = error_points
        self.review_count = review_count
        
    