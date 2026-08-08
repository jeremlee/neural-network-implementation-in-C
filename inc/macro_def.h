#ifndef _MACRO_DEF_H_
#define _MACRO_DEF_H_

#include"defines.h"

#define clip(result) do{ \
    if ((result) > GRADIENT_CLIP) (result) = GRADIENT_CLIP; \
    if ((result) < -GRADIENT_CLIP) (result) = -GRADIENT_CLIP; \
} while(0)


#endif