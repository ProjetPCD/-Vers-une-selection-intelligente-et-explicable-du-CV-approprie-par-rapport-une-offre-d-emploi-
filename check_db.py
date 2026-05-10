import asyncio
import sys
from motor.motor_asyncio import AsyncIOMotorClient

async def check():
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["cv_matcher"]
    print("Jobs in jobs_col:")
    async for j in db["jobs"].find({"_id": "69fe3ae7d3bf3ff798e964a3"}):
        print(j)
    
    print("\nResults in results_col:")
    async for r in db["results"].find({"job_id": "69fe3ae7d3bf3ff798e964a3"}):
        print({k: v for k, v in r.items() if k != 'report'})
        
asyncio.run(check())
