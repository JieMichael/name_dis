## 3月22日-3月27日 改进

* 整体流程分为为三个部分
  1. 从数据库上得到数据并进行初步的cluster生成，dump在本地data文件夹中，代码在read_data_script中，（注：最耗时的一步，十二小时得到1000左右的作者，目前仍在服务器上/home/sjtuiiot/wb/name_dis_3_26/data运行）
  2. 算法通过dump load读取本地的数据，运行（速度很快 在PC上130秒运行了200个人名，运行算法几乎不耗时间）。最终把结果存到data文件夹对应的人名的文件夹内。（新代码在source文件夹内）
  3. 将结果更新数据库（未完成）
* 尝试将其中的dijstra算法改为floyd，更慢了一点，放弃。
* 韩家炜的人名相似算法的代码比较大，目前还在想办法融进，融入后应该能改进很多（未完成）
* 测试算法准确率的部分还未完成