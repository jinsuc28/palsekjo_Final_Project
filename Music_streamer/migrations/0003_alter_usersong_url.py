# Generated by Django 3.2.9 on 2021-11-26 02:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Music_streamer', '0002_usersong_url'),
    ]

    operations = [
        migrations.AlterField(
            model_name='usersong',
            name='url',
            field=models.CharField(default=None, max_length=400),
        ),
    ]