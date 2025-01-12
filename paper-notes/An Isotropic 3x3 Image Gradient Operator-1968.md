We would like to document the derivation of a simple, computationally efficient, gradient operator which we developed in 1968. This operator has been frequently used and referenced since that time. The earliest description of this operator in the computer vision literature is [1], although it has been more widely popularized by its appearance in [2]. Horn [3] defines this operator and references 4 numerical analysis texts [4-7] with the statement:

"Numerical analysis [4-7] teaches us that for certain classes of surfaces an even better estimate is obtained using a weighted average of three such central differences ... These expressions produce excellent estimates for the components of the gradient of the central point". 

The motivation to develop it was to get an efficiently computable gradient estimate which would be more isotropic than the then popular "Roberts Cross" operator [8]. The principle involved is that of estimating the gradient of a digitized picture at a point by the vector summation of the 4 possible simple central gradient estimates obtainable in a ${3\times3}$ neighborhood. The vector summation operation provides an averaging over directions-of-measurement of the gradient. If the density function was truly planar over the neighborhood all 4 gradients would have the same value. Any differences are deviations from local planarity of the function over the neighborhood. The intent here was to extract the direction of the "best" plane although no attempt was made to make this rigorous. 
>  开发 Sobel filter 的动机是得到比 Roberts Cross 算子更加计算高效的各项同性梯度估计算子
>  Sobel filter 的思想是用一个点的 3x3 邻域中获得 4 个可能的中心梯度估计的向量和来估计该点的梯度
>  向量和运算提供了梯度度量方向上的平均，如果密度函数在邻域内是平面的, 则 4 个可能的梯度将具有相同的值，任意差异都是邻域内函数不满足局部平面性的表现，算子的目的是提取出 “最佳” 平面的方向

To be more specific, we will refer here to the image function as a "density" function. (It could just as well be an "intensity" function - the difference depends on the physical nature of the image source.) For a 3x3 neighborhood each simple central gradient estimate is a vector sum of a pair of orthogonal vectors. Each orthogonal vector is a directional derivative estimate multiplied by a unit vector specifying the derivative's direction. The vector sum of these 4 simple gradient estimates amounts to a vector sum of the 8 directional derivative vectors. 
>  称图像函数为密度函数
>  对于一个 3x3 邻域，每个简单的中心梯度估计都是一对正交向量的向量和，每个正交向量都是一个有向的导数估计，即导数估计值乘上表示导数方向的单位向量
>  4 个梯度估计向量的向量和等于 8 个正交向量/有向导数向量的向量和

Thus for a point on a Cartesian grid and its eight neighbors having density values as shown 

$$
\begin{matrix}
\mid\underline a \mid \underline b\mid \underline c\mid\\
\mid\underline d\mid \underline e\mid \underline f\mid\\
\mid\underline g\mid\underline  h \mid\underline  i\mid
\end{matrix}
$$

we define the magnitude of the directional derivative estimate vector 'g' for a given neighbor as 

$$
|g| = \text{<density difference>/<distance to neighbor>}
$$
The direction of 'g' will be given by the unit vector to the appropriate neighbor. 

>  考虑一个邻域如图
>  定义相邻密度值之间的方向导数估计向量 $g$ 的大小为密度差除去邻居之间的距离
>  $g$ 的方向则定义为指向适当邻居的单位向量 (以中心 $e$ 为原点)

Notice that the neighbors group into antipodal pairs: (a,i) (b,h) (c,g) (f,d). Vector summing derivative estimates within each pair causes all the "e" values to cancel leaving the following vector sum for our gradient estimate: 

$$
\begin{align}
G &= (c-g)/4 * [1,1]\\
&\quad +(a-i)/4 *[-1,1]\\
&\quad +(b-h)/2 *[0,1]\\
&\quad + (f-d)/2 *[1,0]
\end{align}
$$ 
>  中心梯度估计是 4 个梯度估计向量的向量和，它可以简化为如上形式

the resultant vector being 

$$
G = [(c-g-a+i)/4 + (f-d)/2, (c-g+a-i)/4 + (b-h)/2]
$$ 
Notice that the square root fortuitously drops out of the formula. If this were to be metrically correct we should divide result by 4 to get the average gradient. 

However, since these operations are typically done in fixed point on small integers and division loses low order significant bits, it is convenient rather to scale the vector by 4, thereby replacing the "divide by 4" (double shift right) with a "multiply by 4" (double shift left) which will preserve low order bits. This leaves us with an estimate which is 16 times as large as the average gradient. The resultant formula is: 

$$
G' = 4*G = [c-g-a+i,2*(f-d), c-g+a-i + 2*(b-h)]
$$ 
>  为了防止除法失去低阶有效位，将 $G$ 再乘上 4 得到 $G'$

It is useful to express this as weighted density summations using the following weighting functions for x and y components: 

$$
\text{x component: }\begin{pmatrix}
 -1 & 0& 1\\
 -2& 0& 2\\
 -1&  0 &  1
\end{pmatrix}\qquad\qquad
\text{y component: }
\begin{pmatrix}
1&2&1\\
0&0&0\\
-1&-2&-1
\end{pmatrix}
$$

>  $G''$ 的计算可以表示为对邻居密度的加权求和，x 部分和 y 部分的权重矩阵如上所示，按照上述权重矩阵加权求和就得到了近似梯度在 x 方向和 y 方向的分量值 $G_x, G_y$

This algorithm was used as an edge point detector in the 1968 vision system [2] at the Stanford Artificial Intelligence Laboratory wherein a point was considered an edge point if and only if 

$$
|G'|^2 > T
$$

where T was a previously chosen threshold. For this purpose it proved an economical alternative to the more robust, but computationally expensive "Hueckel operator" [9]. 

>  Sobel filter 可以用于检测边缘点，认为点的中心梯度估计的长度 $|G| = \sqrt {G_x^2 + G_y^2}$ 超过一定阈值 ($\sqrt T$) 时，该点就是边缘点 (边缘往往是图像特征变化较剧烈的地方，此处的梯度往往不平缓)