device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder_dataset_test = dset.ImageFolder(root='test2/cla2')
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)

net = torch.load('saved_model/siamese.h5').to(device)
test_dataloader = DataLoader(siamese_dataset,batch_size=1,shuffle=True)
dataiter = iter(test_dataloader)
x0,_,_ = next(dataiter)
for i in range(3):
    _,x1,label2 = next(dataiter)
    print(label2)
    concatenated = torch.cat((x0,x1),0)
    
    output1,output2 = net(Variable(x0).to(device),Variable(x1).to(device))
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated),'eula_dis: {:.2f}'.format(euclidean_distance.item()))
