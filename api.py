import math
import random
from typing import Any, Union
from fastapi import FastAPI, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class EucludianFitnessClaculator:
    def __init__(self, depot, orders_dictionary, num_orders):
        self.depot = depot
        self.orders = orders_dictionary
        self.num_orders = num_orders

    def fitness(self, dna):
        last_stop = self.depot
        fitness = 0
        for gene in dna:
            if gene <= self.num_orders + 1:
                cur_stop = self.orders[str(gene)]
                fitness +=  EucludianFitnessClaculator.__distance(last_stop, cur_stop)
                last_stop = cur_stop
            else:
                fitness +=  EucludianFitnessClaculator.__distance(last_stop, self.depot)
                last_stop = self.depot

                
        fitness += EucludianFitnessClaculator.__distance(last_stop, self.depot)
        return fitness

    def __distance(a, b):
        return math.sqrt((a['cx'] - b['cx'])**2 + (a['cy'] - b['cy'])**2)

class GeneticAlgorithmSolver:
    def __init__(self, depot, orders_dictionary, num_orders, num_vehicles = 5, 
        vehicle_max_capacity = 100, max_iter = 1000, mutaion_propability = 0.01, population_size = 1000):
        self.depot = depot
        self.orders = orders_dictionary
        self.num_orders = num_orders
        self.num_vehicles = num_vehicles
        self.vehicle_max_capacity = vehicle_max_capacity
        self.max_iter = max_iter
        self.mutaion_propability = mutaion_propability
        self.fitness_calculator = EucludianFitnessClaculator(self.depot, self.orders, self.num_orders)
        self.population_size = population_size
        self.best_solution = [math.inf, []]
        self.history = []
    
    def generate_random_population(self):
        cur_orders = self.orders.copy()
        routes = []

        for _ in range(self.num_vehicles):
            routes.append([])

        while len(cur_orders) != 0:
            vehicle = random.randint(0,self.num_vehicles-1)
            order_id, _ = random.choice(list(cur_orders.items()))
            del cur_orders[order_id]
            routes[vehicle].append(int(order_id))
        
        population = []
        cur_vehicle = self.num_orders + 2
        for route in routes:
            population += route
            population.append(cur_vehicle)
            cur_vehicle += 1
        population.pop()
        
        return population

    def checkCapacityConstraint(self, dna):
        curCapacity = 0
        for gene in dna:
            if gene <= self.num_orders + 1:
                curCapacity += self.orders[str(gene)]['quantity']
            else:
                curCapacity = 0
            if curCapacity > self.vehicle_max_capacity:
                return False
        if curCapacity > self.vehicle_max_capacity:
                return False
        return True

    def fit(self):
        population = []
        while len(population) < self.population_size:
            dna = self.generate_random_population()
            if self.checkCapacityConstraint(dna):
                population.append((self.fitness_calculator.fitness(dna), dna))

        top = self.generation_evolution(population)
        return top
    
    def generation_evolution(self, population, generation = 1):
        population.sort()

        print('Generation #' + str(generation) + ', fitness: ' + str(population[0][0]))

        if generation >= self.max_iter:
            return
            
        best_solution = population[:150]

        self.history.append(best_solution[0])

        if self.best_solution[0] > best_solution[0][0]:
            self.best_solution = best_solution[0]


        new_population = []
        while len(new_population) < self.population_size:
            parent1 = self.mutation(random.choice(best_solution)[1])
            parent2 = self.mutation(random.choice(best_solution)[1])
            child1, child2 = self.crossover(parent1, parent2)
            if self.checkCapacityConstraint(child1):
                new_population.append((self.fitness_calculator.fitness(child1), child1))
            if self.checkCapacityConstraint(child2):
                new_population.append((self.fitness_calculator.fitness(child2), child2))
        self.generation_evolution(new_population, generation + 1)

    def mutation(self, chromosome):
            
        def inversion_mutation(chromosome_aux):
            chromosome = chromosome_aux
            
            index1 = random.randrange(0,len(chromosome))
            index2 = random.randrange(index1,len(chromosome))
            
            chromosome_mid = chromosome[index1:index2]
            chromosome_mid.reverse()
            
            chromosome_result = chromosome[0:index1] + chromosome_mid + chromosome[index2:]
            
            return chromosome_result
    
        aux = chromosome
        for _ in range(len(chromosome)):
            if random.random() < self.mutaion_propability :
                aux = inversion_mutation(chromosome)
        return aux

    def crossover(self,parent1, parent2):

        def process_gen_repeated(copy_child1,copy_child2):
            count1=0
            for gen1 in copy_child1[:pos]:
                repeat = 0
                repeat = copy_child1.count(gen1)
                if repeat > 1:#If need to fix repeated gen
                    count2=0
                    for gen2 in parent1[pos:]:#Choose next available gen
                        if gen2 not in copy_child1:
                            child1[count1] = parent1[pos:][count2]
                        count2+=1
                count1+=1

            count1=0
            for gen1 in copy_child2[:pos]:
                repeat = 0
                repeat = copy_child2.count(gen1)
                if repeat > 1:#If need to fix repeated gen
                    count2=0
                    for gen2 in parent2[pos:]:#Choose next available gen
                        if gen2 not in copy_child2:
                            child2[count1] = parent2[pos:][count2]
                        count2+=1
                count1+=1

            return [child1,child2]

        pos=random.randrange(1,len(parent1) - 1)
        child1 = parent1[:pos] + parent2[pos:] 
        child2 = parent2[:pos] + parent1[pos:] 
        
        return  process_gen_repeated(child1, child2)

# depot = {"id": "1", "cx": 82.0, "cy": 76.0, "quantity": 0}

# orders_dictionary = {
#     "2": {"id": "2", "cx": 96.0, "cy": 44.0, "quantity": 19.0}, 
#     "3": {"id": "3", "cx": 50.0, "cy": 5.0, "quantity": 21.0}, 
#     "4": {"id": "4", "cx": 49.0, "cy": 8.0, "quantity": 6.0}, 
#     "5": {"id": "5", "cx": 13.0, "cy": 7.0, "quantity": 19.0}, 
#     "6": {"id": "6", "cx": 29.0, "cy": 89.0, "quantity": 7.0}, 
#     "7": {"id": "7", "cx": 58.0, "cy": 30.0, "quantity": 12.0}, 
#     "8": {"id": "8", "cx": 84.0, "cy": 39.0, "quantity": 16.0}, 
#     "9": {"id": "9", "cx": 14.0, "cy": 24.0, "quantity": 6.0}, 
#     "10": {"id": "10", "cx": 2.0, "cy": 39.0, "quantity": 16.0}, 
#     "11": {"id": "11", "cx": 3.0, "cy": 82.0, "quantity": 8.0}, 
#     "12": {"id": "12", "cx": 5.0, "cy": 10.0, "quantity": 14.0}, 
#     "13": {"id": "13", "cx": 98.0, "cy": 52.0, "quantity": 21.0}, 
#     "14": {"id": "14", "cx": 84.0, "cy": 25.0, "quantity": 16.0}, 
#     "15": {"id": "15", "cx": 61.0, "cy": 59.0, "quantity": 3.0}, 
#     "16": {"id": "16", "cx": 1.0, "cy": 65.0, "quantity": 22.0}, 
#     "17": {"id": "17", "cx": 88.0, "cy": 51.0, "quantity": 18.0}, 
#     "18": {"id": "18", "cx": 91.0, "cy": 2.0, "quantity": 19.0}, 
#     "19": {"id": "19", "cx": 19.0, "cy": 32.0, "quantity": 1.0}, 
#     "20": {"id": "20", "cx": 93.0, "cy": 3.0, "quantity": 24.0}, 
#     "21": {"id": "21", "cx": 50.0, "cy": 93.0, "quantity": 8.0}, 
#     "22": {"id": "22", "cx": 98.0, "cy": 14.0, "quantity": 12.0}, 
#     "23": {"id": "23", "cx": 5.0, "cy": 42.0, "quantity": 4.0}, 
#     "24": {"id": "24", "cx": 42.0, "cy": 9.0, "quantity": 8.0}, 
#     "25": {"id": "25", "cx": 61.0, "cy": 62.0, "quantity": 24.0}, 
#     "26": {"id": "26", "cx": 9.0, "cy": 97.0, "quantity": 24.0}, 
#     "27": {"id": "27", "cx": 80.0, "cy": 55.0, "quantity": 2.0}, 
#     "28": {"id": "28", "cx": 57.0, "cy": 69.0, "quantity": 20.0}, 
#     "29": {"id": "29", "cx": 23.0, "cy": 15.0, "quantity": 15.0}, 
#     "30": {"id": "30", "cx": 20.0, "cy": 70.0, "quantity": 2.0}, 
#     "31": {"id": "31", "cx": 85.0, "cy": 60.0, "quantity": 14.0}, 
#     "32": {"id": "32", "cx": 98.0, "cy": 5.0, "quantity": 9.0}}

# num_orders = 32

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Hello World"}

class ModelResponse(BaseModel):
    best_solution: list[Union[float, list[float]]]
    history: Union[list, None] = None

@app.post("/Model/GetResponse", summary="Model Response", response_model=ModelResponse)
def modelResponse(
    depot: dict[str, Any] = Body(description='Depot', default={}),
    orders_dictionary: dict[str, dict[str, Any]] = Body(description='Orders Dictionary', default={}),
    num_orders: int = Body(description='The number of orders', default=32),
    num_drivers: int = Body(description='The number of drivers', default=5),
    with_history: bool = Query(description='Option to return history', default=False),
    history_size: int = Query(description='The length of list history', default=10)
    ):
    solver = GeneticAlgorithmSolver(depot, orders_dictionary, num_orders, num_vehicles=num_drivers, max_iter=600, 
    mutaion_propability=0.03125, population_size=1000)
    solver.fit()

    history: list
    if history_size >= len(solver.history):
        history = solver.history
    else:
        history = solver.history[:history_size]

    if with_history:
        return {
            "best_solution": solver.best_solution,
            "history": history,
        }
    else:
        return {
            "best_solution": solver.best_solution,
        }
