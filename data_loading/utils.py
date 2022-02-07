from sroie.py import sroie_dataloader_train, sroie_dataloader_test
from funsd.py import funsd_dataloader_train, funsd_dataloader_test

def dataloader_train(Dataset):
  if Dataset=='SROIE':
    return(sroie_dataloader_train)
  elif Dataset=='FUNSD':
    return(funsd_dataloader_train)
  
def dataloader_test(Dataset):
  if Dataset=='SROIE':
    return(sroie_dataloader_test)
  elif Dataset=='FUNSD':
    return(funsd_dataloader_test)
  

