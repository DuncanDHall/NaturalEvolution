# Natural Genetics
Serena Chen, Duncan Hall, David Papp, Nathan Yee

### Background and Context

After being introduced to evolutionary algorithms with the SoftDes toolboxes, our group’s goal is to implement a simulation where artificial selection is no longer necessary, as agents reproduce, mutate, and die by themselves, mimicking a form of natural selection. We are creating a self-selecting population of individuals (‘Blobs,’ if you will) that have a certain amount of energy that runs out over time and can be replenished by picking up food. We wish to implement more complex systems after making a successful program that evolves Blobs to get food, such as obstacle or social structures.

### Key Questions

The agents must use a function to determine how to move based on their location and the location of other bodies in the environment, and we are not convinced we have a model which is flexible enough for what we eventually want to do. Currently, agents move by multiplying a vector, including position and velocity data for themselves and their environment, by a matrix, which is unique and represents their D.N.A., producing a vector of accelerations. We are interested in hearing what other methods of behavior implementation come to mind for our audience.

Natural selection simulation is something which we haven’t seen many examples of, and so we’re interested in hearing others’ thoughts on that. We think it’s viable, probably less efficient than artificial selection, but worth exploring even to take advantage of a niche concept. 


### Agenda for Technical Review

We will begin by introducing our program and our objective to the group, and showing a short video of our current implementation (0:00 - 2:00). First we will talk about natural vs. artificial selection, and get everyone’s opinions on that, with post-its as an aid (2:00 - 9:00). Then we will post-it storm, and discuss what the best ways to store the Blob’s attributes and what algorithm to use to translate those attributes to behavior (9:00 - 16:00). After talking about everyone’s suggestions, we will share our current solution and gather feedback on the process we are using (16:00 - 22:00). Then we will go over time (22:00 - 25:00)


