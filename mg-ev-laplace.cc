template<int dim, int spacedim = dim>
class Laplace
{
public:
  Laplace() = default;

  void run();
};

template<int dim, int spacedim>
void
Laplace<dim, spacedim>::run()
{
  // make_grid();
}


int main()
{
  Laplace<2> laplace;
  laplace.run();
}
