if (!requireNamespace("BiocManager", quietly = TRUE))
install.packages("BiocManager")
BiocManager::install("rhdf5")
library(rhdf5)
file_path = 'C:/PIRL/data/MEPOXE0037/Ventilation_Mepo-0037'
key_names <- h5ls(file_path)
print(key_names)
data = h5read(file_path, 'dataArray')
data = aperm(data,c(4,3,2,1))
metadata <- h5readAttributes(file = file_path, '/')
metadata

dev.new(height = 5,width = dim(data)[3])
par(mfrow=c(4,dim(data)[3]),mar=c(0,0,0,0))
for(row in c(1,3,4,5)){
for(col in 1:dim(data)[3]){
im(data[,,col,row],keepPar=1)
}}

