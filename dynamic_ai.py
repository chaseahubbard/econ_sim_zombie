import random
import numpy as np


class DynamicAI:
    def __init__(self, market, good_types):
        self.market = market
        self.goods_list = ['good1', 'good2', 'good3']
        self.trading_radius = 15  # Agents can trade within 15 units of distance
        self.good_types = good_types
    def run_market_cycle(self):
        # Agents update their internal demand
        for agent in self.market.agents:
            agent.update_internal_demand(self.goods_list, self.good_types)
        # Agents attempt to trade with nearby agents
        self.execute_trades()

    def update_internal_demand(self):
        # Define a scaling factor for desired inventory
        scaling_factor = 10

        # Calculate internal demand for each agent
        for agent in self.market.agents:
            agent.update_internal_demand(self.goods_list)
            agent.desired_inventory = {}
            agent.internal_demand = {}
            for good in self.goods_list:
                agent.desired_inventory[good] = agent.values_1.get(good, 0) * scaling_factor
                agent.internal_demand[good] = agent.desired_inventory[good] - agent.inventory.get(good, 0)

    def execute_trades(self):
        # Buyers initiate the trading process
        for buyer in self.market.agents:
            for good in self.goods_list:
                if buyer.internal_demand.get(good, 0) > 0:
                    # Buyer wants to buy this good
                    nearby_agents = buyer.find_nearby_agents(self.market.agents, radius=self.trading_radius)
                    potential_sellers = [a for a in nearby_agents if a != buyer and a.inventory.get(good, 0) > 0]
                    if potential_sellers:
                        buyer_max_price = buyer.values_1.get(good, 0)  # Buyer's maximum acceptable price
                        # Start the auction at buyer's maximum acceptable price
                        current_price = buyer_max_price
                        # Sellers will undercut each other until they reach their minimum acceptable price
                        sellers_in_auction = potential_sellers.copy()
                        selected_seller = None  # Initialize selected_seller
                        while True:
                            # Each seller decides if they are willing to offer at the current price
                            willing_sellers = []
                            for seller in sellers_in_auction:
                                seller_min_price = seller.values_1.get(good, 0)  # Seller's minimum acceptable price
                                if current_price >= seller_min_price:
                                    willing_sellers.append(seller)
                            if not willing_sellers:
                                # No sellers willing to sell at this price
                                break
                            if len(willing_sellers) == 1:
                                # Only one seller willing to sell at this price
                                selected_seller = willing_sellers[0]
                                break
                            else:
                                # Multiple sellers willing to sell, reduce price slightly
                                current_price -= 0.01
                                # Ensure price does not go below the lowest seller's minimum acceptable price
                                min_seller_price = min(s.values_1.get(good, 0) for s in willing_sellers)
                                if current_price < min_seller_price:
                                    current_price = min_seller_price
                            # Check if price cannot be reduced further
                            if current_price <= 0:
                                break
                        if selected_seller is None:
                            # No seller was selected, move to next good or buyer
                            continue
                        # Determine the maximum quantity buyer can afford at this price
                        if current_price == 0:
                            continue  # Prevent division by zero
                        max_affordable_quantity = buyer.wealth // current_price
                        # Determine the quantity to trade
                        quantity = min(
                            buyer.internal_demand[good],
                            selected_seller.inventory[good],
                            max_affordable_quantity
                        )
                        quantity = int(quantity)
                        if quantity >= 1:
                            total_price = current_price * quantity
                            # Execute trade
                            buyer.spend(total_price)
                            selected_seller.earn(total_price)
                            selected_seller.inventory[good] -= quantity
                            buyer.inventory[good] = buyer.inventory.get(good, 0) + quantity
                            # Update internal demand
                            selected_seller.internal_demand[good] += quantity  # Seller's demand decreases
                            buyer.internal_demand[good] -= quantity
                            # Print transaction details
                            print(f"\nTransaction executed between {selected_seller.name} and {buyer.name}:")
                            print(f"{selected_seller.name} sold {quantity} units of {good} to {buyer.name} at {current_price:.2f} per unit.")
                            print("Current Market State:")
                            print(self.market)
                        else:
                            continue  # Quantity less than 1, skip
                    else:
                        # No potential sellers
                        continue

    def execute_trade(self, buyer, seller, good, price):
        # Determine the maximum quantity buyer can afford at this price
        if price == 0:
            return  # Prevent division by zero
        max_affordable_quantity = buyer.wealth // price
        # Determine the quantity to trade
        quantity = min(
            buyer.internal_demand[good],
            seller.inventory[good],
            max_affordable_quantity
        )
        quantity = int(quantity)
        if quantity >= 1:
            total_price = price * quantity
            # Execute trade
            buyer.spend(total_price)
            seller.earn(total_price)
            seller.inventory[good] -= quantity
            buyer.inventory[good] = buyer.inventory.get(good, 0) + quantity
            # Update internal demand
            seller.internal_demand[good] += quantity  # Seller's demand decreases
            buyer.internal_demand[good] -= quantity
            # Print transaction details
            print(f"\nTransaction executed between {seller.name} and {buyer.name}:")
            print(f"{seller.name} sold {quantity} units of {good} to {buyer.name} at {price:.2f} per unit.")
            print("Current Market State:")
            print(self.market)
