from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt


# Create your views here.
from PIL import Image

@csrf_exempt
def inference(request):
  if request.method == 'POST' and request.FILES.get('image'):
    image_file = request.FILES['image']
    image = Image.open(image_file)
  return JsonResponse({'error': 'Invalid request'}, status=400)
