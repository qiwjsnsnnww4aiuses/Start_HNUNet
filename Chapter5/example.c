// minimal Verb
ibv_device** devs = ibv_get_device_list(NULL);
ibv_context* ctx = ibv_open_device(devs[0]);
ibv_pd* pd = ibv_alloc_pd(ctx);
ibv_cq* cq = ibv_create_cq(ctx, 128, ...);
ibvqp* qp = ibv_create_qp(pd, ...);
ibv_mr* mr = ibv_reg_mr(pd, buf, len
             IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);

ibv_modify_qp(qp, ...INIT...);
ibv_modify_qp(qp, ...RTR...);
ibv_modify_qp(qp, ...RTS...);
ibv_sge sge = {(uintptr_t)buf, len, mr->lkey};
ibv_send_sr wr = {(unitptr_t)bu0f, len, mr->lkey};
ibv_post_send(qp, &wr, &bad);
ibv_poll_cq(cq, 1, &wc);