Tensorboard visualization

```
$ user@local_host: ssh -L port_local:127:0.0.1:port_remote user@remote_host
```

```
$ user@remote_host: tensorboard --logdir=runs --port port_remote
```
This loads outputs from ./runs/ and visualize on localhost:port_local. 
