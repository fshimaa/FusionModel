import csv
import matplotlib.pyplot as plt
import os
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    # print(file_path)
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size





def readfiles(filename):
    my_file = open  ( filename , "r" )
    valacc=[]
    valloss=[]
    # reading the file
    data = my_file.read ( )
    data_into_list = data.split ( "\n" )
    for i in range ( 1 , 26 ):
        print ( data_into_list[ i ] )
        f = data_into_list[ i ].split ( "\t" )
        Accuracy = f[ 2 ]  # Accuracy
        valacc.append ( float ( Accuracy ) )
        loss = f[ 1 ].split ( ',' )[ 0 ].split ( "(" )[ 1 ]  # loss
        valloss.append(float(loss))
    my_file.close ( )
    return valloss,valacc


def DrawLoss( train_loss,val_loss,epochs=25 ):

    plt.figure()
    # plt.title("Training and Validation Loss")
    plt.rc ( 'font' , weight='bold' )
    plt.rcParams.update ( {'font.size': 12} )
    plt.plot(val_loss,label="Valid",color="blue")
    plt.plot(train_loss,label="Train", color="orange")
    plt.xlabel("Epoch Number",fontsize=12,fontweight='bold')
    plt.ylabel("Loss",fontsize=12,fontweight='bold')
    plt.legend( loc= "upper right")
    plt.savefig(os.path.join('/home/data/class_output/', 'C_Loss.png'),dpi=300)

def DrawAccur(tain_acur ,val_acur,epochs=25):
    plt.figure()
    # plt.title("Training and Validation Accuracy")
    plt.rc ( 'font' , weight='bold' )
    plt.rcParams.update ( {'font.size': 12} )
    plt.plot(val_acur,label="Valid",color="blue")
    plt.plot(tain_acur,label="Train", color="orange")
    plt.xlabel("Epoch Number",fontsize=12,fontweight='bold')
    plt.ylabel("Accuracy",fontsize=12,fontweight='bold')
    plt.legend ( loc="lower right" )
    # plt.show()
    plt.savefig(os.path.join('/home/data/class_output/', 'C_Accuracy.png'),dpi=300)


