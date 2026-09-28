# 08 · BlockPool 与核心数据结构（W2-D8）

## 核心问题
1. KVCacheBlock / FreeKVCacheBlockQueue / BlockHashToBlockMap 的设计与不变量？
2. touch / free / evict 的精确语义？
3. null block 的地位？

## 代码索引
- `vllm/v1/core/block_pool.py`（135 起，方法全表见 00-code-map §1）
- `vllm/v1/core/kv_cache_utils.py` KVCacheBlock(176)/FreeKVCacheBlockQueue(247)

## 种子结论
- FreeKVCacheBlockQueue 是侵入式双向链表（块自带 prev/next_free_block 指针 + fake head/tail）：
  O(1) popleft、O(1) 中间 remove（touch/重插需要），这是 deque 做不到的。
- touch = ref_cnt++ 且从 free 队列摘除；free = ref_cnt-- 归零后按「淘汰优先序」入队：
  hashed 块 append 到 tail（LRU，最后被复用，保命中率），unhashed 块 append 到 head
  （LIFO，优先复用刚释放的物理页，GPU 局部性好）。free_blocks 要求调用方按 reverse
  allocation order 传块——搞错会静默劣化命中率。
- get_new_blocks 从 head 弹块；若弹出的是有 hash 的块，先清 hash（可被外部重建）并发 BlockRemoved。
- BlockHashToBlockMap：单块直存，同 hash 多块（CoW 保留期/partial hit）时退化为 {block_id: block}。
- null block：池初始化时创建，ref_cnt 不维护、永不 free/hash；窗口回收与 skip 位置用它占位。

## 沉淀区
