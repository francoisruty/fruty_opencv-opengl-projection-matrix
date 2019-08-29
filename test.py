import cv2
import numpy as np
import math

cx = 88 #principal point x coord
cy = 109 #principal point y coord
w = 178 #image width
h = 218 #image height
near = 1 #near plane
far = 10 #far plane
fovy = 45.0/360.0*2.0*np.pi #45° in radians
f = h / 2*math.tan(fovy/2) #cf http://paulbourke.net/miscellaneous/lens/

#we compute the OpenCV camera matrix
camera_mtx = np.array([
[f, 0, cx],
[0., f, cy],
[0.,0.,1.]
], dtype=np.float64)


#we compute the corresponding opengl projection matrix
#cf https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
#NOTE: K00 = K11 = f, K10 = 0.0, K02 = cx, K12 = cy, K22 = 1.0
opengl_mtx = np.array([
[2*f/w, 0.0, (w - 2*cx)/w, 0.0],
[0.0, -2*f/h, (h - 2*cy)/h, 0.0],
[0.0, 0.0, (-far - near) / (far - near), -2.0*far*near/(far-near)],
[0.0, 0.0, -1.0, 0.0]
])


#point is in opencv camera space (along Oz axis)
point = np.array([1.0, 2.0, 8.0]) #Note: coords must be floats


#### OpenCV projection
screen_point, _ = cv2.projectPoints(np.array([point]), np.zeros(3), np.zeros(3), camera_mtx, np.zeros(5))
print(screen_point)

#Note: we obtain the same result with this: (that's what cv2.projectPoints basically does: multiply points with camera matrix and then divide result by z coord)
print(camera_mtx.dot(point)/point[2])


#### OpenGL projection
#we flip the point z coord, because in opengl camera is oriented along -Oz axis
point[2] = -point[2]
point2 = np.hstack([point,1.0]) #we add vertex w coord (usually done in vertex shader before multiplying by projection matrix)
#we get the point in clip space
clip_point = opengl_mtx.dot(point2)
#NOTE: what follows "simulates" what happens in OpenGL after the vertex shader.
#This is necessary so that we can make sure our projection matrix will yield the correct result when used in OpenGL
#we get the point in NDC
ndc_point = clip_point / clip_point[3]
#we get the screen coordinates
viewport_point = (ndc_point + 1.0)/2.0 * np.array([w, h, 1.0, 1.0])
#opencv Oy convention is opposite of OpenGL so we reverse y coord
viewport_point[1] = h - viewport_point[1]
print(viewport_point)

#Now you can see that viewport_point and screen_point have the same x/y coordinates!
#This means you can now, from OpenCv camera matrix, use OpenGl to render stuff on top of the image,
#thanks to the opengl projection matrix, computed from opencv camera matrix
