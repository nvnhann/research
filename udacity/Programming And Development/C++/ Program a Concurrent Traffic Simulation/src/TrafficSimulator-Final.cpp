#include <iostream>
#include <thread>
#include <vector>
#include <algorithm> // for std::for_each

#include "Vehicle.h"
#include "Street.h"
#include "Intersection.h"
#include "Graphics.h"

// Function to create traffic objects for Paris
void createTrafficObjects_Paris(std::vector<std::shared_ptr<Street>> &streets, std::vector<std::shared_ptr<Intersection>> &intersections, std::vector<std::shared_ptr<Vehicle>> &vehicles, std::string &filename, int nVehicles)
{
    filename = "../data/paris.jpg";

    int nIntersections = 9;
    for (size_t ni = 0; ni < nIntersections; ni++)
    {
        intersections.push_back(std::make_shared<Intersection>());
    }

    std::vector<std::pair<int, int>> intersectionPositions = {
        {385, 270}, {1240, 80}, {1625, 75}, {2110, 75},
        {2840, 175}, {3070, 680}, {2800, 1400}, {400, 1100},
        {1700, 900}};

    for (size_t i = 0; i < intersectionPositions.size(); i++)
    {
        intersections[i]->setPosition(intersectionPositions[i].first, intersectionPositions[i].second);
    }

    int nStreets = 8;
    for (size_t ns = 0; ns < nStreets; ns++)
    {
        streets.push_back(std::make_shared<Street>());
        streets.at(ns)->setInIntersection(intersections.at(ns));
        streets.at(ns)->setOutIntersection(intersections.at(8));
    }

    for (size_t nv = 0; nv < nVehicles; nv++)
    {
        vehicles.push_back(std::make_shared<Vehicle>());
        vehicles.at(nv)->setCurrentStreet(streets.at(nv));
        vehicles.at(nv)->setCurrentDestination(intersections.at(8));
    }
}

// Function to create traffic objects for NYC
void createTrafficObjects_NYC(std::vector<std::shared_ptr<Street>> &streets, std::vector<std::shared_ptr<Intersection>> &intersections, std::vector<std::shared_ptr<Vehicle>> &vehicles, std::string &filename, int nVehicles)
{
    filename = "../data/nyc.jpg";

    int nIntersections = 6;
    for (size_t ni = 0; ni < nIntersections; ni++)
    {
        intersections.push_back(std::make_shared<Intersection>());
    }

    std::vector<std::pair<int, int>> intersectionPositions = {
        {1430, 625}, {2575, 1260}, {2200, 1950}, {1000, 1350},
        {400, 1000}, {750, 250}};

    for (size_t i = 0; i < intersectionPositions.size(); i++)
    {
        intersections[i]->setPosition(intersectionPositions[i].first, intersectionPositions[i].second);
    }

    int nStreets = 7;
    for (size_t ns = 0; ns < nStreets; ns++)
    {
        streets.push_back(std::make_shared<Street>());
        streets.at(ns)->setInIntersection(intersections.at(ns % nIntersections)); // loop the streets based on the number of intersections
        streets.at(ns)->setOutIntersection(intersections.at((ns + 1) % nIntersections)); // loop to next intersection
    }

    for (size_t nv = 0; nv < nVehicles; nv++)
    {
        vehicles.push_back(std::make_shared<Vehicle>());
        vehicles.at(nv)->setCurrentStreet(streets.at(nv % nStreets)); // loop vehicles to streets
        vehicles.at(nv)->setCurrentDestination(intersections.at(nv % nIntersections)); // loop vehicles to intersections
    }
}

// Function to simulate traffic objects
void simulateTrafficObjects(std::vector<std::shared_ptr<Intersection>> &intersections, std::vector<std::shared_ptr<Vehicle>> &vehicles)
{
    std::for_each(intersections.begin(), intersections.end(), [](std::shared_ptr<Intersection> &i)
                  { i->simulate(); });

    std::for_each(vehicles.begin(), vehicles.end(), [](std::shared_ptr<Vehicle> &v)
                  { v->simulate(); });
}

// Function to launch visualization
void launchVisualization(const std::string &backgroundImg, const std::vector<std::shared_ptr<Intersection>> &intersections, const std::vector<std::shared_ptr<Vehicle>> &vehicles)
{
    std::vector<std::shared_ptr<TrafficObject>> trafficObjects;

    std::for_each(intersections.begin(), intersections.end(), [&trafficObjects](std::shared_ptr<Intersection> intersection)
                  { trafficObjects.push_back(std::dynamic_pointer_cast<TrafficObject>(intersection)); });

    std::for_each(vehicles.begin(), vehicles.end(), [&trafficObjects](std::shared_ptr<Vehicle> vehicle)
                  { trafficObjects.push_back(std::dynamic_pointer_cast<TrafficObject>(vehicle)); });

    Graphics graphics;
    graphics.setBgFilename(backgroundImg);
    graphics.setTrafficObjects(trafficObjects);
    graphics.simulate();
}

int main()
{
    std::vector<std::shared_ptr<Street>> streets;
    std::vector<std::shared_ptr<Intersection>> intersections;
    std::vector<std::shared_ptr<Vehicle>> vehicles;
    std::string backgroundImg;
    int nVehicles = 6;

    createTrafficObjects_Paris(streets, intersections, vehicles, backgroundImg, nVehicles);

    simulateTrafficObjects(intersections, vehicles);

    launchVisualization(backgroundImg, intersections, vehicles);

    return 0;
}