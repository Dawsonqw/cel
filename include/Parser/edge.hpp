#ifndef BASE_EDGE_H
#define BASE_EDGE_H
namespace cel{
    class Edge{
        public:
            Edge()=default;
            virtual ~Edge()=default;

            Edge(const Edge&)=delete;
            Edge& operator=(const Edge&)=delete;

            virtual void set_weight(float weight)=0;
            virtual float get_weight()=0;
    };
}
#endif