from tensorflow import pad
from tensorflow.keras.layers import Layer



"""
based on https://www.machinecurve.com/index.php/2020/02/10/using-constant-padding-reflection-padding-and-replication-padding-with-keras/#
"""
class ReplicationPadding2D(Layer):    
    """
    Args:
        padding (tuple): paddings to use, in form (top, bottom, left, right)
    """

    def __init__(self, padding=(1, 0, 0, 0), **kwargs):        
        self.padding = tuple(padding)  
        super(ReplicationPadding2D, self).__init__(**kwargs)    
    
    def compute_output_shape(self, input_shape):        
        return (input_shape[0], 
                input_shape[1] + self.padding[0] + self.padding[1], 
                input_shape[2] + self.padding[2] + self.padding[3], 
                input_shape[3])
    
    def call(self, input_tensor, mask=None):        
        top, bottom, left, right = self.padding        
        return pad(
            input_tensor, 
            [ [0,0], [top, bottom], [left, right], [0,0] ], 
            'SYMMETRIC'
        )
