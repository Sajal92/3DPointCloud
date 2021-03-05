import numpy as np
import laspy as lp
import pptk

def preparedata():
    input_path = "/home/bee/PycharmProjects/test/"
    dataname = "NZ19_Wellington"
    point_cloud=lp.file.File(input_path+dataname+".las", mode="r")
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    colors = np.vstack((point_cloud.red, point_cloud.green,point_cloud.blue)).transpose()
    #normals = np.vstack((point_cloud.normalx, point_cloud.normaly,point_cloud.normalz)).transpose()
    return point_cloud,points,colors#,normals

def pptkviz(points,colors):
    v = pptk.viewer(points)
    v.attributes(colors/65535)
    v.set(point_size=0.001,bg_color= [0,0,0,0],show_axis=0,show_grid=0)
    return v

def cameraSelector(v):
    camera=[]
    camera.append(v.get('eye'))
    camera.append(v.get('phi'))
    camera.append(v.get('theta'))
    camera.append(v.get('r'))
    return np.concatenate(camera).tolist()

def computePCFeatures(points, colors, knn=10, radius=np.inf):
    normals=pptk.estimate_normals(points,knn,radius)
    idx_ground=np.where(points[...,2]>np.min(points[...,2]+0.3))
    idx_normals=np.where(abs(normals[...,2])<0.9)
    idx_wronglyfiltered=np.setdiff1d(idx_ground, idx_normals)
    common_filtering=np.append(idx_normals, idx_wronglyfiltered)
    return points[common_filtering],colors[common_filtering]

if __name__ == "__main__":
    point_cloud,points,colors=preparedata()
    viewer1=pptkviz(points,colors)

cam1=cameraSelector(v)
#Change your viewpoint then -->
cam2=cameraSelector(v)
#Change your viewpoint then -->
cam3=cameraSelector(v)
#Change your viewpoint then -->
cam4=cameraSelector(v)
poses = []
poses.append(cam1)
poses.append(cam2)
poses.append(cam3)
poses.append(cam4)
v.play(poses, 2 * np.arange(4), repeat=True, interp='linear')