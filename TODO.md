i like a newtunnelmanager 

as the old one is not working right

we know that  ssh -L 19555:udc-an26-1:19555 uva -N works

a) make sure i can specifiy local and remote port. by default remote is the same as local
b) make sure the default host is taken dynamically from the node on which it runs. for example cmc llm status reports it correctly
c) make sure ther is a decorator that for each functin in the new tunnel manager 
   prints the function name and its parameters when it is called

do nt yet integrate it intot orchetsrtor

