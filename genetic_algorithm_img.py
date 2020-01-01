#!/usr/bin/env python
# coding: utf-8

# In[1]:


from random import uniform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import random, pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
import cv2, time


DEBUG = True


def save_pickle(data, filename):
    with open("{}.pickle".format(filename), 'wb') as t:
        pickle.dump(data, t)
    print("PICKLE SAVE DONE")

def show_img(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)

def load_pickle(filepath):
    with open("{}.pickle".format(filepath), 'rb') as t:
        data = pickle.load(t)
    print("PICKLE LOAD DONE")
    return data


# ## Configs

# In[3]:


# chormosome의 개수
POPULATION = 100

# 좋은 chormosome의 개수
GOOD_PARENTS_CNT = 30

# 운좋은 chormosome의 개수
LUCKY_PARENTS_CNT = 10

# 각 parent가 만드는 chromosome 개수
CHILDREN_CNT = 5

# 돌연변이 생성 비율
MUTATION_PROB = 0.1

# 최대 세대 수 
MAX_GENERATIONS = 50


# ## Load the image

# In[4]:


# Load Image

IMA_ARR = cv2.imread('Bigger-Splash-1967.jpg')

# jpg_img_arr = mpimg.imread('Bigger-Splash-1967.jpg')
# jpg_IMG = Image.open('Bigger-Splash-1967.jpg')
# FLATTENED = jpg_img_arr.flatten()
# LEN_FLATTENED = len(FLATTENED)
# print("flattend size :", FLATTENED.shape)


class chromoSome:
    def __init__(self, real_img, res=10, max_shapes=10, chromo_data=None):
        self.real_img = real_img
        self.img_size = real_img.shape
        self.res = res
        self.max_shapes = max_shapes
        
        if chromo_data is None:
            self.create_random_img()
        else:
            self.img = chromo_data
    
    def __repr__(self):
        return "chromosome fitenss : {}".format(self.fitness)
    
    @property
    def fitness(self):
        # score가 0에 가까우면 좋은 것
        
        score = 0
        
        #dist = abs(self.real_img-self.img).sum()
        dist = np.linalg.norm(self.real_img.astype('float') - self.img.astype('float')) / (self.img_size[0] * self.img_size[1])
        score = 1. / dist
        
        return score
    
    def create_random_img(self):
        self.img = np.zeros(self.img_size, np.uint8)
        self.assign_circle()
        
        
    def assign_circle(self):
        # center_x = np.random.randint(0, self.img_size[1])
        # center_y = np.random.randint(0, self.img_size[0])
        # radius = np.random.randint(0, self.img_size[0]/4)
        # #radius   = np.random.randint(0, int(self.img_size[0] / (1.1*self.res)))
        # opacity  = np.random.rand(1)[0]
        # color    = chromoSome.get_bgr_color()
        
        overlay  = self.img.copy()
        n_shapes = np.random.randint(0, self.max_shapes)
        
        for _ in range(n_shapes):
            center_x = np.random.randint(0, self.img_size[1])
            center_y = np.random.randint(0, self.img_size[0])
            radius = np.random.randint(0, self.img_size[0]/4)
            #radius   = np.random.randint(0, int(self.img_size[0] / (1.1*self.res)))
            opacity  = np.random.rand(1)[0]
            color    = chromoSome.get_bgr_color()
            cv2.circle(overlay, (center_x, center_y), radius, color, -1)
            cv2.addWeighted(overlay, opacity, self.img, 1 - opacity, 0, self.img)
        
        
    def get_bgr_color():
        blue  = np.random.randint(0, 255)
        green = np.random.randint(0, 255)
        red   = np.random.randint(0, 255)
        return (blue, green, red)
        

class Generation:
    cnt = 0
    def __init__(self, population):
        Generation.cnt += 1
        self.generation_lv = Generation.cnt
        self.population = population
        self.sorted_pop = self.sort_pop()

    
    def __repr__(self):
        return "Generation Level : {}".format(self.generation_lv)
    
 
    def assign_fitness(self):
        self.best_chromo = self.sorted_pop[0]
        self.second_chromo = self.sorted_pop[1]
           
    def evolution(self):
        
        #print("Start Evolution Generation level %d" % Generation.cnt)
        self.assign_fitness()
        

        mutation_cnt = POPULATION * MUTATION_PROB
        #good_parents_cnt = POPULATION - mutation_parents_cnt

        children = list()
        for num in range(POPULATION):
            child = self.make_child()
            if num >= POPULATION - mutation_cnt:
                child = self.make_mutation(child)
            children.append(child)
                
        return Generation(children)
    
    def sort_pop(self):
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        #self.best = sorted_pop[0]
        
        return sorted_pop
    
    def make_child(self):

        ind1_weight = np.random.rand(1)[0]
        new_image = np.zeros((self.population[0].img_size), dtype=np.uint8)
        
        cv2.addWeighted(self.best_chromo.img, ind1_weight, self.second_chromo.img, 1 - ind1_weight, 0, new_image)
        child = chromoSome(real_img=self.best_chromo.real_img, chromo_data=new_image)

        # if DEBUG:
        #     cv2.imshow("title",child.img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        return child
    
    def make_mutation(self, child):
        overlay  = child.img.copy()
        n_shapes = np.random.randint(0, 10)
        
        for _ in range(n_shapes):
            center_x = np.random.randint(0, child.img.shape[1])
            center_y = np.random.randint(0, child.img.shape[0])
            radius = np.random.randint(0, child.img.shape[0]/4)
            opacity  = np.random.rand(1)[0]
            color    = chromoSome.get_bgr_color()
            cv2.circle(overlay, (center_x, center_y), radius, color, -1)
            cv2.addWeighted(overlay, opacity, child.img, 1 - opacity, 0, child.img)
        
        return child
    
    @property
    def mean_fitness(self):
        return np.mean([chromosome.fitness for chromosome in self.population])
    
    @property
    def get_best(self):
        return sorted(self.population, key=lambda x: x.fitness, reverse=True)[0]

    
    def fitness(self):
        return np.mean([chromo.fitness for chromo in self.population])


# In[7]:


if __name__ == "__main__":
    # cv2.imshow("title",IMA_ARR)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    IMA_ARR = cv2.imread('Bigger-Splash-1967.jpg')

    fitness_list = list()
    initial_pop = [chromoSome(real_img=IMA_ARR) for _ in range(POPULATION)]
    gen = Generation(initial_pop)


    
    if DEBUG:
        print(gen.sorted_pop[1])
        print(gen.sorted_pop[99])
        #print(initial_pop)

    # cv2.imshow("title",gen.sorted_pop[0].img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow("title",gen.get_best.img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow("title",gen.sorted_pop[50].img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

 
    for i in tqdm(range(10000)):
        gen = gen.evolution()
        best = gen.get_best
        fitness_list.append(best.fitness)
        
        if i % 20 == 0:
            cv2.imwrite("./img/{}_img.jpg".format(i), best.img)
            

