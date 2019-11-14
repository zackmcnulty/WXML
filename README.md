# WXML
Washington Experimental Mathematics Laboratory (WXML) is a computational-oriented mathematics research lab targeted for undergraduates.
My teams project this quarter focused on matching complexes. A matching on a graph G is some subset (possibly empty) of the edges of G such that
no edges in the subset are adjacent (share a vertex). The matching complex is the collection of all these subsets and is in turn a simplicial complex.
Simplicial complexes are more abstract mathematical structures studied in a more general setting than what we did here, but to put it simply a simplicial
complex on a set X is a subset of the powerset of X that is closed under subsets. Namely, if S is a simplicial complex, and x is in S, then so are all subsets of x 
(remember x is an element of the powerset of X and hence a set itself).

You can find more information on [wikipedia](https://en.wikipedia.org/wiki/Simplicial_complex).


Lots of our team's inspiration came from this [paper](https://arxiv.org/abs/1906.03328). 



### Code Instructions

First, start off by cloning this repository using git. Essentially, you just need to download git (software) from the internet
and run `git clone https://github.com/zackmcnulty/WXML.git` in the terminal/command-line. This will create a folder in your local
directory and you can access the code there. Also run `pip install -r requirements.txt` in the command line to download all the 
python libraries I use in this code.

I have tried to make this code as accessible as possible. The main functions provided here is findind/drawing matching complex
given a graph G AND finding a graph that generates a given matching complex. All of the primary functions are found in the file
`matching_complexes.py` but I have also created some helper scripts `find_complex_from_graph.py` and `find_graph_from_complex.py` 
which have the obvious uses. In these helper files, you just need to specify some needed information (e.g. the edge list for the graph or
the matching complex) and run them: they will do the rest. 

If you want more flexibility in what you do, take a look at `matching_complexes.py`. You can import its methods into another python file
and use them as you wish.

Side note: I also created a script `linegraph.py` that plots the line graph of a given graph (specified in the file itself) if you find this useful.
