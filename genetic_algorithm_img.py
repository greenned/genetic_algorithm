import numpy as np
import random, pickle
from tqdm import tqdm
import cv2, time, argparse




# 돌연변이 생성 비율
MUTATION_PROB = 0.1


class chromoSome:
    def __init__(self, real_img, max_shapes=100, chromo_data=None):
        self.real_img = real_img
        self.img_size = real_img.shape
        self.max_shapes = max_shapes
        
        if chromo_data is None:
            self.create_random_img()
        else:
            self.img = chromo_data
    
    def __repr__(self):
        return "chromosome fitenss : {}".format(self.fitness)
    
    @property
    def fitness(self):
        score = 0
        dist = np.linalg.norm(self.real_img.astype('float') - self.img.astype('float')) / (self.img_size[0] * self.img_size[1])
        
        # 높을수록 좋음
        score = 1. / dist
        
        return score
    
    def create_random_img(self):
        self.img = np.zeros(self.img_size, np.uint8)
        random_assign = random.choice([chromoSome.assign_circle, chromoSome.assign_polygon])
        random_assign(self.img, self.max_shapes, self.img_size)
        # self.assign_circle()
        
        
    def assign_circle(img, max_shapes, img_size):
        overlay  = img.copy()
        n_shapes = np.random.randint(0, max_shapes)
        
        for _ in range(n_shapes):
            center_x = np.random.randint(0, img_size[1])
            center_y = np.random.randint(0, img_size[0])
            radius = np.random.randint(0, img_size[0]/4)
            #radius   = np.random.randint(0, int(img_size[0] / (1.1*res)))
            opacity  = np.random.rand(1)[0]
            color    = chromoSome.get_bgr_color()
            cv2.circle(overlay, (center_x, center_y), radius, color, -1)
            cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

    def assign_polygon(img, max_shapes, img_size):
        pts = []
        point = [np.random.randint(0, img_size[1]), np.random.randint(0, img_size[0])]
        pts.append(point)

        n_shapes = np.random.randint(0, max_shapes)

        for i in range(n_shapes):
            new_point = [point[0] + np.random.randint(-1, 2) * np.random.randint(0, int(img_size[1] / 4)), point[1] + np.random.randint(-1, 2) * np.random.randint(0, int(img_size[0] / 4))]
            pts.append(new_point)


        pts 	 = np.array(pts)
        pts 	 = pts.reshape((-1, 1, 2))
        opacity  = np.random.rand(1)[0]
        color    = chromoSome.get_bgr_color()


        overlay  = img.copy()
        cv2.fillPoly(overlay, [pts], color, 8)
        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0 ,img)

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
        # 최상위 Fitness 유전자 설정
        self.assign_fitness()
        
        # 돌연변이 갯수 설정
        n_population = len(self.population)
        mutation_cnt = n_population * MUTATION_PROB

        children = list()
        for num in range(n_population):
            # 상위 1,2위로 자식을 만듬
            child = self.make_child()
            # 만든 자식들 중에 돌연변이를 만듬(circle을 덧댐)
            if num < mutation_cnt:
                child = self.make_mutation(child)
            children.append(child)
                
        return Generation(children)
    
    def sort_pop(self):
        # 유전자 Fitness 좋은 순으로 정렬
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        return sorted_pop
    
    def make_child(self):
        # 랜덤으로 첫번째 opacity 비중설정
        ind1_weight = np.random.rand(1)[0]
        new_image = np.zeros((self.population[0].img_size), dtype=np.uint8)
        
        cv2.addWeighted(self.best_chromo.img, ind1_weight, self.second_chromo.img, 1 - ind1_weight, 0, new_image)
        child = chromoSome(real_img=self.best_chromo.real_img, chromo_data=new_image)

        return child
    
    def make_mutation(self, child):
        overlay  = child.img.copy()
        n_shapes = np.random.randint(1, 10)

        random_assign = random.choice([chromoSome.assign_circle, chromoSome.assign_polygon])
        random_assign(child.img, n_shapes, child.img.shape)
        
        # for _ in range(n_shapes):
        #     center_x = np.random.randint(0, child.img.shape[1])
        #     center_y = np.random.randint(0, child.img.shape[0])
        #     radius = np.random.randint(0, child.img.shape[0]/4)
        #     opacity  = np.random.rand(1)[0]
        #     color    = chromoSome.get_bgr_color()
        #     cv2.circle(overlay, (center_x, center_y), radius, color, -1)
        #     cv2.addWeighted(overlay, opacity, child.img, 1 - opacity, 0, child.img)
        
        return child
    
    @property
    def mean_fitness(self):
        return np.mean([chromosome.fitness for chromosome in self.population])
    
    @property
    def get_best(self):
        return sorted(self.population, key=lambda x: x.fitness, reverse=True)[0]
   
    def fitness(self):
        return np.mean([chromo.fitness for chromo in self.population])
    

def save_pickle(data, filename):
    with open("{}.pickle".format(filename), 'wb') as t:
        pickle.dump(data, t)
    print("PICKLE SAVE DONE")

def load_pickle(filepath):
    with open("{}.pickle".format(filepath), 'rb') as t:
        data = pickle.load(t)
    print("PICKLE LOAD DONE")
    return data


def main(file_path, n_population, n_generation):
    IMA_ARR = cv2.imread(file_path)
    fitness_list = list()
    initial_pop = [chromoSome(real_img=IMA_ARR) for _ in range(n_population)]
    gen = Generation(initial_pop)

    try:
        for i in tqdm(range(n_generation)):
            
                gen = gen.evolution()
                best = gen.get_best
                fitness_list.append(best.fitness)

                if i % 100 == 0:
                    cv2.imwrite("/Users/greenned/result/유전알고리즘/picaso_with_circle_and_polygon/{}_img.jpg".format(i), best.img)
    finally:
        save_pickle(fitness_list, "./pickle/fitness_list")
        save_pickle(gen, "./pickle/last_generation")


if __name__ == "__main__":

    DEBUG = False
    if DEBUG:
        file_path = "picaso.png"
        n_population = 100
        n_generation = 100000
        main(file_path, n_population, n_generation)
    
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("-np", "--n_pop", required=False, type=int)
        ap.add_argument("-ng", "--n_gen", required=False, type=int)
        ap.add_argument("-path", "--file_path", required=True, type=str)

        args = vars(ap.parse_args())

        if args["n_pop"] != None:
            n_population = args["n_pop"]

        if args['n_gen'] != None:
            n_generation = args['n_gen']

        if args['file_path'] != None:
            file_path = args['file_path']
        
        main(file_path, n_population, n_generation)
