from typing import Any, Union
from fastapi import FastAPI, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from solver import GeneticAlgorithmSolver

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
