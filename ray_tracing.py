# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#функция нормирования вектора
def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflected(vector, normal): #функция рассчета отражения
    return vector - 2 * np.dot(vector, normal) * normal


def nearest_intersected_object(objects, ray_origin, ray_direction): #ищем ближайший объект и дистанцию до него(объект, который лучш пересечет первым)
    distances = np.ones(len(objects))

    for index in range(len(objects)):

        if objects[index]['surface'] == 'sphere':
            b = 2 * np.dot(ray_direction, ray_origin - objects[index]['center'])
            c = np.linalg.norm(ray_origin - objects[index]['center']) ** 2 - objects[index]['radius'] ** 2
            delta = b ** 2 - 4 * c
            if delta > 0:
                t1 = (-b + np.sqrt(delta)) / 2
                t2 = (-b - np.sqrt(delta)) / 2
                if t1 > 0 and t2 > 0:
                    distances[index] = min(t1,t2)
                else:
                    distances[index] = None
            else:
                distances[index] = None

        if objects[index]['surface'] == 'plane':
            t = (np.dot(objects[index]['point'], objects[index]['normal']) - np.dot(ray_origin,objects[index]['normal']))/ (np.dot(ray_direction,objects[index]['normal']))
            if t > 0:
                distances[index] = t
            else:
                distances[index] = None

    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance

width = 1080 #ширина сцены в пикселях
height = 920 #высота сцены в пикселях
max_depth = 3

camera = np.array([0, 0, 1]) #положение камеры
ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio) # left, top, right, bottom

light = { 'position': np.array([5, 5, 5]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }

objects = [
{ 'surface' : 'sphere',  'center': np.array([-0.4, 0, -1]), 'radius': 0.8, 'ambient': np.array([0.7, 0.5, 0]), 'diffuse': np.array([0.7, 0, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
{ 'surface' : 'sphere',  'center': np.array([0.2, -0.3, 0]), 'radius': 0.3, 'ambient': np.array([0.2, 0, 0.1]), 'diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
{ 'surface' : 'sphere',  'center': np.array([-0.55, 0.5, 0]), 'radius': 0.15, 'ambient': np.array([0, 0.1, 0]), 'diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 50, 'reflection': 0.5 },
{ 'surface' : 'plane',  'point': np.array([0, -0.7, 0]), 'normal': np.array([0, 1, 0]), 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 }
] #задаем объекты сцены

image = np.zeros((height, width, 3)) #матрицей нулей задаем изначальные значения цвета для каждого пикселя
for i, y in enumerate(np.linspace(screen[1], screen[3], height)): #в цикле определяем значение цвета для каждого пикселя(RGB)
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):

        pixel = np.array([x, y, 0])
        origin = camera
        direction = normalize(pixel - origin)
        color = np.zeros((3)) #задаем цвет как три нуля - черный цвет в RGB
        reflection = 1

        for k in range(max_depth):

            nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
            if nearest_object is None:
                break

            intersection = origin + min_distance * direction
            if(nearest_object['surface'] == 'sphere'):
                normal_to_surface = normalize(intersection - nearest_object['center'])
            if (nearest_object['surface'] == 'plane'):

                if np.dot(nearest_object['normal'],direction ) > 0:
                    normal_to_surface = -1 * nearest_object['normal']
                else:
                    normal_to_surface = nearest_object['normal']

            shifted_point = intersection + 1e-5 * normal_to_surface
            intersection_to_light = normalize(light['position'] - shifted_point)

            _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)

            intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
            is_shadowed = min_distance < intersection_to_light_distance

            if is_shadowed: #если в тени - то пиксель черный
                break

            illumination = np.zeros((3))
            illumination += nearest_object['ambient'] * light['ambient']
            illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light, normal_to_surface)
            intersection_to_camera = normalize(camera - intersection)
            H = normalize(intersection_to_light + intersection_to_camera)
            illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4)
            color += reflection * illumination
            reflection *= nearest_object['reflection']
            origin = shifted_point
            direction = reflected(direction, normal_to_surface)

        image[i, j] = np.clip(color, 0, 1)
    print("%d/%d" % (i + 1, height))
plt.imsave('image.png', image) #сохраняем изображение