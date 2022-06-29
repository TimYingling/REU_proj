from torch.utils.data import DataLoader, ConcatDataset
from data.synthia import Synthia

def setup_loaders(dataset, path_list, batch_size):
    combined_data = []
    for i in range(len(dataset)):
        if(dataset[i] == 'rellis3d'):
            data = relis3d.Relis3D(path_list[i])
            data_loader = DataLoader(data, batch_size=batch_size)
            if (len(dataset) == 1):
                return data_loader
            else:
                combined_data.append(data)
                del data_loader
                del data
        if(dataset[i] == 'semantickitti'):
            data = semantickitti.SemanticKitti(path_list[i])
            data_loader = DataLoader(data, batch_size=batch_size)
            if (len(dataset) == 1):
                return data_loader
            else:
                combined_data.append(data)
                del data_loader
                del data
        if(dataset[i] == 'cityscapes'):
            data = cityscapes.CityScapes(path_list[i])
            data_loader = DataLoader(data, batch_size=batch_size)
            if (len(dataset) == 1):
                return data_loader
            else:
                combined_data.append(data)
                del data_loader
                del data

        if(dataset[i] == 'ourdataset'):
            data = ourdataset.Dataset(path_list[i])
            data_loader = DataLoader(data, batch_size=batch_size)
            if (len(dataset) == 1):
                return data_loader
            else:
                combined_data.append(data)
                del data_loader
                del data
                
        if(dataset[i] == 'synthia'):
            data = Synthia(path_list[i])
            data_loader = DataLoader(data, batch_size=batch_size)
            if(len(dataset) == 1):
                return data_loader
            else:
                combined_data.append(data)
                del data_loader
                del data

    data_loader = DataLoader(
        ConcatDataset(combined_data),
        batch_size=batch_size, shuffle=True)
    return data_loader