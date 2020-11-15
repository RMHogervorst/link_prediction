# Objective: Build a link prediction model to predict future links (mutual
# likes) between unconnected nodes (Facebook pages).
library(igraph)
library(tidymodels)
nodes <- readr::read_csv("data/fb-pages-food.nodes")
edges <- readr::read_csv("data/fb-pages-food.edges",col_names = c("from","to")) 
range(nodes$new_id) # 0-619
range(edges$from) # 0-618
range(edges$to) # 0-619
nodes <- nodes %>% 
  select(id = new_id, name) %>% 
  mutate(
    id = id +1,
    name=paste0(name," (",id,")")
    )
edges <- edges %>% 
  mutate(from= from+1, to=to+1)
### Tidygraph (or igraph to be precise) cannot deal with negative or 0 indexed
### edge information, so I add 1
g <-tidygraph::tbl_graph(edges=edges,nodes = nodes,directed = FALSE)

# visualize graph for a bit ## ----
library(ggraph)
ggraph(g)+
  geom_node_point(color = "lightblue")+
  geom_edge_link(alpha = 1/3)+
  labs(subtitle = "Very often just plotting your data leads to a hairball plot")

## dataset preperation for model building ----

# for every node, find the nodes within 2 hops.

# find negative examples, non existing edges between nodes.
# using an adjacency matrix



# rather ugly solution to make this adj list again.
dist_g <- distances(g )
#diag(dist_g) %>% sum() #diagonal is 0 
#
# find all nodes within 2 distance
# reshape2::melt(dist_g) %>% filter(value ==0) %>% filter(Var1 != Var2) # is empty
distances <- 
  reshape2::melt(dist_g, value.name = "distance") %>% 
  filter(Var1 != Var2) %>% # throws away all 0s(direct edges)
  left_join(nodes, by=c("Var1"="name")) %>% 
  rename(to=id) %>% 
  left_join(nodes, by=c("Var2"="name")) %>% 
  rename(from=id) 

distances %>% filter(from==49) %>% filter(distance==1)
# checked in gephi. the distances are correct.
# node 49 is connected to?
# g %>% tidygraph::activate(edges) %>% filter(from == 49 | to == 49) %>% as.data.frame() 
# 32 358 519 525 165
# edges %>% filter(from==49) %>% pull(to) # 519 525 165
# edges %>% filter(to==49) %>% pull(from) # 32 358
distances %>% filter(from == 49 | to ==49) %>% filter(distance ==1)
# only 329
distances %>% filter(from==49) %>% filter(to==519) # distance 5!
distances %>% filter(to==49) %>% filter(from==32) # distance 5!
distances %>% filter(from==49) %>% filter(to==165) # distance 5!

# blue ribbon restaurants(474) with josh marks (287)
# edges %>% filter(to == 474, from == 287) (result 0, is not there)
g %>% 
  mutate(color_ =ifelse(id %in% c(287, 474),"terminus",NA_character_)) %>% 
  morph(to_shortest_path, from=287, to=474) %>% 
  mutate(color_= 'path') %>% 
  activate(edges) %>% 
  mutate(color_='path') %>% 
  unmorph() %>% 
  ggraph()+
  geom_edge_link(aes(color = color_))+
  geom_node_point(aes(color = color_))


non_connected <- as.data.frame(as.table(adj_m == 0 & adj_m <=2))


result <- 
  non_connected %>% 
  filter(!Freq) %>% 
  select(-Freq) %>% 
  left_join(nodes, by=c("Var1"="name")) %>% 
  rename(to=id) %>% 
  left_join(nodes, by=c("Var2"="name")) %>% 
  rename(from=id) %>% 
  filter(to != from) %>% 
  # make sure current edges are excluded
  anti_join(edges, by=c("from","to")) %>% 
  anti_join(edges, by=c("from"="to","to"="from"))

# odly enough, I have more than python version.
# python version has 19.018 unconnected
# I have 215085, but they do shortes path length within 2
anti_join(result, edges) %>% nrow()

# negative examples are 2 or less distance and not 1.
neg_examples <- distances %>% filter(distance ==2)
### AFter finding negative examples
library(progress)
pb <- progress_bar$new(total = nrow(edges))
set.seed(12445)
idx <- sample(1:nrow(edges),size = nrow(edges),replace = FALSE)
edges <- edges[idx,]
pos_examples_idx <- rep(FALSE, nrow(edges))
g_temp <- g
# Go through edges one by one
for (row in seq_len(nrow(edges))) {
  # check if graph is still connected
  # it so add to list.
  pb$tick()
  g_temp1 <- g_temp %>% 
    activate(edges) %>% 
    anti_join(edges[row,], by=c("from","to"))
  verdict <- with_graph(g_temp1, graph_is_connected())
  pos_examples_idx[row] <- verdict
  # if not skip it
  if(verdict){g_temp <- g_temp1}  
}
message(paste0("Found ",sum(pos_examples_idx), " possible links"))
# list of edges will be positive examples
positive_examples <- edges[pos_examples_idx,] 
nrow(positive_examples) # 1617
nrow(neg_examples) # 34306   # ~ 21 times more.

#### Collect info over every edge.
# give every positive example a 1 
# give every negative example a 0
trainingset <- 
  bind_rows(
  positive_examples %>% mutate(target=1),
  neg_examples[,c("from","to")] %>% mutate(target=0)
)

# drop removable links 
emptier_graph <- g %>% 
  activate(edges) %>% 
  anti_join(positive_examples, by = c("from", "to"))
stopifnot(with_graph(emptier_graph, graph_is_connected()))

### feature creation
## node2vec?

node_features <- emptier_graph %>% 
  activate(nodes) %>% 
  mutate(
    # to be honest, I'm doing random things here
    degree= tidygraph::centrality_degree(normalized=TRUE),
    betweenness = tidygraph::centrality_betweenness(cutoff = 5,normalized = TRUE),#
    pg_rank = centrality_pagerank(),
    eigen = centrality_eigen(),
    br_score= node_bridging_score() # takes quite long
    ) %>% 
  as_tibble() %>% 
  select(-name)

enriched_trainingset <-
  trainingset %>% 
  left_join(node_features, by=c("from"="id")) %>% 
  left_join(node_features, by=c("to"="id"), suffix=c("","_to"))

# start with
## usemodels::use_glmnet(enriched_trainingset, target~., verbose = FALSE, tune=FALSE, prefix = "ntwrk")

ntwrk_recipe <-
  recipe(enriched_trainingset,formula = target~.) %>% 
  recipes::update_role(to, new_role = "other") %>% 
  recipes::update_role(from, new_role = "other") %>% 
  step_interact(terms = ~ pg_rank:pg_rank_to) %>% 
  step_interact(terms = ~ degree:degree_to) %>% 
  step_corr(all_numeric()) %>% 
  step_nzv(all_predictors()) %>% 
  step_normalize(all_predictors(), -all_nominal()) %>% 
  step_mutate(target = as.factor(target))
#rec_network %>% prep()

ntwrk_spec <- 
  logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet")

ntwrk_workflow <- 
  workflow() %>% 
  add_recipe(ntwrk_recipe) %>% 
  add_model(ntwrk_spec) 

### split into training and test set
set.seed(2345)
tr_te_split <- initial_split(enriched_trainingset,strata = target)
val_set <- validation_split(training(tr_te_split),strata = target, prop = .8)
## Setting up tune grid manually, because it is just one column
lr_reg_grid <- tibble(penalty = 10^seq(-4, -1, length.out = 30))

ntwrk_res <-
  ntwrk_workflow %>% 
  tune_grid(val_set,
            grid = lr_reg_grid,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc, bal_accuracy,f_meas))

## visualise results
lr_plot <- 
  ntwrk_res %>% 
  collect_metrics() %>% 
  ggplot(aes(x = penalty, y = mean)) + 
  geom_point() + 
  geom_line() + 
  scale_x_log10(labels = scales::label_number())+
  facet_grid(.metric~.,scales = "free_y")+
  labs(
    title = "Slightly problematic curves here"
  )

ntwrk_res %>% 
  collect_metrics() %>% 
  filter(.metric == "roc_auc") %>% 
  ggplot(aes(x = penalty, y = mean)) + 
  geom_point() + 
  geom_line() + 
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number())

## show best models
top_models <-
  ntwrk_res %>% 
  show_best("roc_auc", n = 5) %>% 
  arrange(penalty)

lr_best <- 
  ntwrk_res %>% 
  collect_metrics() %>% 
  arrange(penalty) %>% 
  slice(5)

pred_auc <- 
  ntwrk_res %>% 
  collect_predictions(parameters = lr_best) %>% 
  roc_curve(target, .pred_0) %>% 
  mutate(model = "Logistic Regression")
autoplot(pred_auc)

### it is not really that good. probably different model
## for example random forest


### what is the performance on the actual testset?
### 
### replace model with parameters set (still untrained)
### top_models[1,]
ntwrk_spec_1 <- 
  logistic_reg(penalty = 0.003562248, mixture = 1) %>% 
  set_engine("glmnet")
## change model
updated_workflow <- 
  ntwrk_workflow %>% 
  update_model(ntwrk_spec_1)

ntwrk_fit <- 
  updated_workflow %>% 
  last_fit(tr_te_split)

ntwrk_fit %>% 
  collect_metrics()

ntwrk_fit %>% 
  collect_predictions() %>% 
  roc_curve(target, .pred_0) %>% 
  autoplot()

# NOW WE COULD BRING IT BACK AROUND AND PLOT PREDICTED VALUES
# https://bgreenwell.github.io/pdp/articles/pdp-classification.html
# 
# OR USE THIS MODEL
# CAN WE EXTRACT THE MODEL AND PREDICT ON NEW DATA?
# ON NEW INCOMING DATA WITH THE SAME FORM?

#### used the following resources
# * https://www.tidymodels.org/start/case-study/ for the model setup
# * https://www.analyticsvidhya.com/blog/2020/01/link-prediction-how-to-predict-your-future-connections-on-facebook/ for the setup, choice of data and selection criteria
# * network data facebook pages http://networkrepository.com/fb-pages-food.php#
# ( no attribution for facebook here?)
# @inproceedings{nr,
# title={The Network Data Repository with Interactive Graph Analytics and Visualization},
# author={Ryan A. Rossi and Nesreen K. Ahmed},
# booktitle={AAAI},
# url={http://networkrepository.com},
# year={2015}
# }